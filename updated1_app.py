import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update
import dash_cytoscape as cyto
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx, remove_self_loops
from torch_geometric.explain import Explainer, GNNExplainer

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import traceback
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Global Variables & Setup ---
MODEL_DIR = 'models'
DATA_DIR = 'data'
DEFAULT_DATASET = 'Cora'
DEFAULT_MODEL = 'GCN'
AVAILABLE_DATASETS = ['Cora', 'CiteSeer']
AVAILABLE_MODELS = ['GCN', 'GAT']

# --- Path Management ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
MODEL_ABS_DIR = os.path.join(SCRIPT_DIR, MODEL_DIR)
DATA_ABS_DIR = os.path.join(SCRIPT_DIR, DATA_DIR)

# --- GPU/CPU Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- GNN Explainer App ---")
print(f"Using device: {device}")

# --- Model Definitions ---
class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout_rate=0.6, **kwargs):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- Data Loading & Model Instantiation ---
def load_precomputed_package(model_type, dataset_name):
    filename = f"{model_type}_{dataset_name}_tuned.pkl"
    path = os.path.join(MODEL_ABS_DIR, filename)
    if not os.path.exists(path):
        print(f"ERROR: Pre-computed file not found: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            package = pickle.load(f)
        return package
    except Exception as e:
        print(f"ERROR: Failed to load pre-computed file '{path}'. Error: {e}")
        return None

def instantiate_model(package, model_type, num_features, num_classes):
    if not package: return None
    try:
        model_args = package['model_init_args']
        if model_type == 'GCN': model = GCNNet(in_channels=num_features, out_channels=num_classes, **model_args)
        elif model_type == 'GAT': model = GATNet(in_channels=num_features, out_channels=num_classes, **model_args)
        else: raise ValueError(f"Unknown model type: {model_type}")
        
        state_dict = {key: torch.tensor(value) if isinstance(value, list) else value for key, value in package['model_state_dict'].items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"ERROR: Could not instantiate model from package. Error: {e}")
        return None

# --- Explanation & Inference ---
@torch.no_grad()
def run_inference(model, data):
    model.eval()
    log_probs = model(data.x.to(device), data.edge_index.to(device))
    return log_probs.argmax(dim=-1).cpu().numpy()

@torch.no_grad()
def run_cpa_iv(model, data, node_idx, top_k=5, max_path_len=2):
    model.eval()
    data = data.to(device)
    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, max_path_len, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    sub_node_idx = torch.where(subset == node_idx)[0].item()
    sub_data = Data(x=data.x[subset], edge_index=sub_edge_index).to(device)
    log_probs = model(sub_data.x, sub_data.edge_index)
    original_probs = torch.exp(log_probs)
    predicted_class = original_probs[sub_node_idx].argmax()
    baseline_score = original_probs[sub_node_idx, predicted_class].item()
    nx_graph = to_networkx(Data(edge_index=remove_self_loops(sub_data.edge_index)[0], num_nodes=sub_data.num_nodes), to_undirected=False)
    paths = [path for source in nx_graph.nodes() if source != sub_node_idx for path in nx.all_simple_paths(nx_graph, source=source, target=sub_node_idx, cutoff=max_path_len) if len(path) > 1]
    path_effects = []
    for path in paths[:100]:
        perturbed_edge_index = sub_data.edge_index.clone()
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            mask = ~((perturbed_edge_index[0] == u) & (perturbed_edge_index[1] == v))
            perturbed_edge_index = perturbed_edge_index[:, mask]
        perturbed_log_probs = model(sub_data.x, perturbed_edge_index)
        perturbed_score = torch.exp(perturbed_log_probs)[sub_node_idx, predicted_class].item()
        path_effects.append({'path': [subset[node].item() for node in path], 'score': baseline_score - perturbed_score})
    return sorted(path_effects, key=lambda x: x['score'], reverse=True)[:top_k]

# --- Visualization Helpers ---
def create_subgraph_cytoscape(data, center_node_idx, predictions, class_to_color_map, explanation, cpa_data):
    if not (0 <= center_node_idx < data.num_nodes): return [], set()
    
    subset, sub_edge_index, _, _ = k_hop_subgraph(center_node_idx, 2, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    nodes_in_subgraph = set(subset.cpu().numpy())
    
    explained_edges = set()
    if explanation and explanation.get('edge_mask'):
        edge_mask = torch.tensor(explanation['edge_mask'])
        important_indices = torch.where(edge_mask > 0.1)[0]
        if len(important_indices) > 0 and important_indices.max() < data.edge_index.shape[1]:
            for idx in important_indices:
                u, v = data.edge_index[0, idx].item(), data.edge_index[1, idx].item()
                explained_edges.add(tuple(sorted((u, v))))

    causal_edges = set()
    if cpa_data:
        for p_info in cpa_data:
            path = p_info['path']
            for i in range(len(path) - 1):
                causal_edges.add(tuple(sorted((path[i], path[i+1]))))

    cyto_elements = []
    for node_id in nodes_in_subgraph:
        pred_class_idx = int(predictions[node_id])
        cyto_elements.append({
            'data': {'id': str(node_id), 'label': str(node_id)}, 
            'style': {'background-color': class_to_color_map.get(pred_class_idx, '#808080')}, 
            'classes': 'center-node' if node_id == center_node_idx else ''
        })

    processed_edges = set()
    for i in range(sub_edge_index.shape[1]):
        u, v = sub_edge_index[0, i].item(), sub_edge_index[1, i].item()
        edge_tuple = tuple(sorted((u, v)))
        if edge_tuple in processed_edges: continue
        processed_edges.add(edge_tuple)
        classes = []
        if edge_tuple in explained_edges: classes.append('explained-edge')
        if edge_tuple in causal_edges: classes.append('causal-path-edge')
        cyto_elements.append({'data': {'source': str(u), 'target': str(v)}, 'classes': ' '.join(classes)})
        
    return cyto_elements, nodes_in_subgraph

def plot_embeddings(embeddings_2d, predictions, true_labels, selected_node_id, class_to_color_map, num_classes):
    if embeddings_2d is None: return go.Figure(layout={'title': "Embeddings Not Available"})
    df = pd.DataFrame(embeddings_2d, columns=['Dim 1', 'Dim 2'])
    df['Node ID'] = [str(i) for i in range(len(df))]
    df['Predicted Class'] = predictions.astype(str)
    df['True Label'] = true_labels
    
    category_orders = {"Predicted Class": [str(i) for i in range(num_classes)]}
    
    fig = px.scatter(df, x='Dim 1', y='Dim 2', color='Predicted Class', 
                     hover_data=['Node ID', 'True Label'], custom_data=['Node ID'],
                     color_discrete_map={str(k):v for k,v in class_to_color_map.items()},
                     category_orders=category_orders)
    
    fig.update_layout(
        title={
            'text': "Node Embeddings",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend_title_text=None,
        clickmode='event+select', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=50, l=10, r=10)
    )
    return fig

def plot_2hop_importance(data, explanation, node_idx):
    if explanation is None or explanation.get('edge_mask') is None:
        return go.Figure(layout={'title': "2-Hop Importance (Select Node)"})

    edge_mask_np = np.array(explanation['edge_mask'])
    edge_index = torch.tensor(data.edge_index)
    
    if edge_index.shape[1] != len(edge_mask_np):
        return go.Figure(layout={'title': "2-Hop Importance (Error: Mismatch)"})

    subset, _, _, _ = k_hop_subgraph(node_idx, 2, edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    nodes_in_subgraph = set(subset.cpu().numpy())
    
    edge_to_score = {tuple(edge): score for edge, score in zip(edge_index.t().tolist(), explanation['edge_mask'])}

    node_importance = {}
    for n_id in nodes_in_subgraph:
        if n_id == node_idx: continue
        mask = (edge_index[0] == n_id) | (edge_index[1] == n_id)
        incident_edges = edge_index[:, mask].t().tolist()
        max_score = 0
        for edge in incident_edges:
            max_score = max(max_score, edge_to_score.get(tuple(edge), 0))
        if max_score > 0:
            node_importance[n_id] = max_score

    if not node_importance: return go.Figure(layout={'title': f"No important neighbors for Node {node_idx}"})
    
    df = pd.DataFrame(node_importance.items(), columns=['Node ID', 'Max Edge Importance']).sort_values('Max Edge Importance', ascending=False)
    fig = px.bar(df, x='Node ID', y='Max Edge Importance', title=f'2-Hop Node Importance for Node {node_idx}', text_auto='.2f')
    fig.update_layout(xaxis_type='category', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Dash App ---
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
app.title = "Interactive GNN Explainer"

STYLESHEET = [
    {'selector': 'node', 'style': {'label': 'data(label)', 'font-size': '14px', 'text-valign': 'center', 'color': 'white', 'text-outline-color': '#2c3e50', 'text-outline-width': 2, 'shape': 'ellipse', 'width': 35, 'height': 35}},
    {'selector': '.center-node', 'style': {'shape': 'star', 'background-color': '#f1c40f', 'width': 50, 'height': 50}},
    {'selector': 'edge', 'style': {'width': 2, 'line-color': '#bdc3c7', 'curve-style': 'bezier'}},
    {'selector': '.explained-edge', 'style': {'line-color': '#2ecc71', 'width': 6, 'opacity': 0.9}},
    {'selector': '.causal-path-edge', 'style': {'line-color': '#e74c3c', 'width': 5, 'line-style': 'dashed'}},
]

app.layout = html.Div(style={'maxWidth': '1800px', 'margin': 'auto'}, children=[
    dcc.Store(id='precomputed-package-store'),
    dcc.Store(id='full-dataset-store'),
    dcc.Store(id='editable-graph-store'),
    dcc.Store(id='current-predictions-store'),
    dcc.Store(id='selected-node-store'),
    dcc.Store(id='explanation-store'),
    dcc.Store(id='cpa-iv-store'),
    dcc.Store(id='status-message-store', data=None),

    html.Div(className='row', style={'padding': '10px 0', 'display': 'flex', 'alignItems': 'center'}, children=[
        html.Div(className='four columns', children=[html.H1("Interactive GNN Explainer", style={'fontSize': '4rem', 'margin': 0})]),
        html.Div(className='three columns', children=[dcc.Input(id='select-node-input', type='number', placeholder='Select Node by ID...', min=0, step=1, style={'width': '100%'})]),
        html.Div(className='three columns', children=[dcc.Dropdown(id='dataset-dropdown', options=AVAILABLE_DATASETS, value=DEFAULT_DATASET, clearable=False)]),
        html.Div(className='two columns', children=[dcc.Dropdown(id='model-dropdown', options=AVAILABLE_MODELS, value=DEFAULT_MODEL, clearable=False)]),
    ]),
    html.Div(id='status-bar'),

    html.Div(className='row', style={'padding': '5px 0'}, children=[
        html.Div(className='five columns', children=[
            html.H4("Graph View", style={'marginTop': 0, 'marginBottom': '5px'}),
            cyto.Cytoscape(id='graph-view', stylesheet=STYLESHEET, layout={'name': 'cose'}, style={'width': '100%', 'height': '230px', 'border': '1px solid #ddd'}),
        ]),
        html.Div(className='three columns', children=[
            html.H4("Neighborhood Details", style={'marginTop': 0, 'marginBottom': '5px'}),
            html.Div(id='subgraph-legend-view', style={'height': '230px', 'overflowY': 'auto', 'border': '1px solid #ddd', 'padding': '10px'})
        ]),
        html.Div(className='four columns', children=[
            html.H4("Embeddings", style={'marginTop': 0, 'marginBottom': '5px'}),
            dcc.Graph(id='embeddings-view', style={'height': '230px'})
        ]),
    ]),
    
    html.Div(className='row', style={'marginTop': '20px'}, children=[
        html.H4("Detailed Analysis & Editing"),
        dcc.Tabs(id='analysis-tabs', value='tab-explainer', children=[
            dcc.Tab(label='GNNExplainer', value='tab-explainer', children=[
                dcc.Graph(id='neighbor-importance-view', style={'height': '280px'}),
            ]),
            dcc.Tab(label='Causal (CPA-IV)', value='tab-cpa', children=[
                html.Button('Run Causal Path Analysis', id='run-cpa-iv-button', n_clicks=0, style={'marginTop': '10px', 'width': '100%'}),
                html.Div(id='cpa-iv-output', style={'padding': '10px'}),
            ]),
            dcc.Tab(label='Attention (GAT)', value='tab-attention', children=[html.Div(id='attention-view', style={'padding': '10px'})]),
            dcc.Tab(label='Edit Graph', value='tab-edit', children=[
                html.Div(style={'padding': '10px'}, children=[
                    html.Strong("Add/Remove Edge"),
                    dcc.Input(id='edge-source-input', type='number', placeholder='Source Node', style={'marginRight': '5px'}),
                    dcc.Input(id='edge-target-input', type='number', placeholder='Target Node', style={'marginRight': '5px'}),
                    html.Button('Add Edge', id='add-edge-button', n_clicks=0, style={'marginRight': '5px'}),
                    html.Button('Remove Edge', id='remove-edge-button', n_clicks=0),
                ])
            ]),
        ]),
    ])
])

# --- Callbacks ---

@callback(Output('status-bar', 'children'), Output('status-bar', 'style'), Input('status-message-store', 'data'))
def update_status_bar(status_data):
    if not status_data: return "", {'display': 'none'}
    text, msg_type = status_data.get('text', ''), status_data.get('type', 'info')
    colors = {'info': '#d1ecf1', 'success': '#d4edda', 'error': '#f8d7da'}
    style = {'padding': '10px', 'borderRadius': '5px', 'marginBottom': '10px', 'backgroundColor': colors.get(msg_type, '#d1ecf1')}
    return text, style

@callback(
    Output('precomputed-package-store', 'data'), Output('full-dataset-store', 'data'),
    Output('editable-graph-store', 'data'), Output('current-predictions-store', 'data'),
    Input('dataset-dropdown', 'value'), Input('model-dropdown', 'value')
)
def load_initial_data(dataset_name, model_type):
    package = load_precomputed_package(model_type, dataset_name)
    if package is None: raise dash.exceptions.PreventUpdate
    from torch_geometric.datasets import Planetoid
    dataset_obj = Planetoid(root=DATA_ABS_DIR, name=dataset_name)
    data = dataset_obj[0]
    full_dataset = {'x': data.x.tolist(), 'y': data.y.tolist(), 'num_nodes': data.num_nodes, 'num_features': data.num_features, 'num_classes': dataset_obj.num_classes, 'name': dataset_name}
    editable_graph = {'edge_index': data.edge_index.tolist()}
    initial_predictions = {'preds': package['predictions']}
    return package, full_dataset, editable_graph, initial_predictions

@callback(
    Output('editable-graph-store', 'data', allow_duplicate=True),
    Output('status-message-store', 'data', allow_duplicate=True),
    Input('add-edge-button', 'n_clicks'), Input('remove-edge-button', 'n_clicks'),
    State('edge-source-input', 'value'), State('edge-target-input', 'value'),
    State('editable-graph-store', 'data'), State('full-dataset-store', 'data'),
    prevent_initial_call=True
)
def update_edges(add_clicks, remove_clicks, source, target, graph_data, dataset):
    if not all([source is not None, target is not None, graph_data, dataset]): raise dash.exceptions.PreventUpdate
    if not (0 <= source < dataset['num_nodes'] and 0 <= target < dataset['num_nodes']):
        return no_update, {'text': 'Node ID out of range.', 'type': 'error'}

    edge_index = graph_data['edge_index']
    triggered_id = ctx.triggered_id
    status = no_update
    
    if triggered_id == 'add-edge-button':
        if ([source, target] not in list(zip(edge_index[0], edge_index[1]))):
             edge_index[0].extend([source, target])
             edge_index[1].extend([target, source])
             status = {'text': f'Added edge {source}-{target}. Re-running model...', 'type': 'info'}
    
    elif triggered_id == 'remove-edge-button':
        new_edge_index = [[], []]
        removed = False
        for u, v in zip(edge_index[0], edge_index[1]):
            if not ((u == source and v == target) or (u == target and v == source)):
                new_edge_index[0].append(u)
                new_edge_index[1].append(v)
            else:
                removed = True
        edge_index = new_edge_index
        if removed: status = {'text': f'Removed edge {source}-{target}. Re-running model...', 'type': 'info'}

    return {'edge_index': edge_index}, status

@callback(
    Output('current-predictions-store', 'data', allow_duplicate=True),
    Input('editable-graph-store', 'data'),
    State('full-dataset-store', 'data'),
    State('precomputed-package-store', 'data'),
    State('model-dropdown', 'value'),
    prevent_initial_call=True
)
def update_predictions_on_edit(graph_data, dataset, package, model_type):
    if not all([graph_data, dataset, package]): raise dash.exceptions.PreventUpdate
    model = instantiate_model(package, model_type, dataset['num_features'], dataset['num_classes'])
    if model is None: raise dash.exceptions.PreventUpdate
    
    pyg_data = Data(x=torch.tensor(dataset['x'], dtype=torch.float), edge_index=torch.tensor(graph_data['edge_index'], dtype=torch.long))
    new_preds = run_inference(model, pyg_data)
    
    return {'preds': new_preds.tolist()}

@callback(Output('selected-node-store', 'data'), Input('graph-view', 'tapNodeData'), Input('embeddings-view', 'clickData'), Input('select-node-input', 'value'))
def update_selected_node(tapNode, clickData, input_value):
    trigger_id, node_id = ctx.triggered_id, None
    if trigger_id == 'graph-view' and tapNode: node_id = tapNode['id']
    elif trigger_id == 'embeddings-view' and clickData: node_id = clickData['points'][0]['customdata'][0]
    elif trigger_id == 'select-node-input' and input_value is not None: node_id = str(input_value)
    return {'id': node_id} if node_id is not None else no_update

@callback(
    Output('explanation-store', 'data'),
    Input('selected-node-store', 'data'),
    Input('editable-graph-store', 'data'),
    State('precomputed-package-store', 'data'), State('full-dataset-store', 'data'),
    State('current-predictions-store', 'data'), State('model-dropdown', 'value'),
    prevent_initial_call=True
)
def run_gnn_explainer(selected_node, graph_data, package, dataset, predictions_data, model_type):
    if not all([selected_node, package, dataset, graph_data, predictions_data]): return no_update
    node_idx = int(selected_node['id'])
    model = instantiate_model(package, model_type, dataset['num_features'], dataset['num_classes'])
    if model is None: return no_update
    
    x = torch.tensor(dataset['x'], dtype=torch.float, device=device)
    edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long, device=device)
    target = torch.tensor([predictions_data['preds'][node_idx]], device=device)

    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=100), explanation_type='model', edge_mask_type='object', model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs'))
    explanation = explainer(x=x, edge_index=edge_index, target=target, index=node_idx)
    return {'edge_mask': explanation.get('edge_mask').cpu().tolist() if explanation.get('edge_mask') is not None else None}

@callback(
    Output('cpa-iv-store', 'data'),
    Output('status-message-store', 'data', allow_duplicate=True),
    Input('run-cpa-iv-button', 'n_clicks'),
    State('selected-node-store', 'data'),
    State('precomputed-package-store', 'data'),
    State('full-dataset-store', 'data'),
    State('editable-graph-store', 'data'),
    State('model-dropdown', 'value'),
    prevent_initial_call=True
)
def run_cpa_iv_callback(n_clicks, selected_node, package, dataset, graph_data, model_type):
    if n_clicks == 0 or not selected_node: return no_update, no_update
    node_idx = int(selected_node['id'])
    
    model = instantiate_model(package, model_type, dataset['num_features'], dataset['num_classes'])
    if model is None: return no_update, {'text': "CPA-IV failed: Could not load model.", "type": "error"}
    
    pyg_data = Data(x=torch.tensor(dataset['x'], dtype=torch.float), edge_index=torch.tensor(graph_data['edge_index'], dtype=torch.long), num_nodes=dataset['num_nodes'])
    causal_paths = run_cpa_iv(model, pyg_data, node_idx)
    status = {'text': f"CPA-IV found {len(causal_paths)} influential paths for Node {node_idx}.", "type": "success"}
    return causal_paths, status

@callback(
    Output('graph-view', 'elements'), Output('embeddings-view', 'figure'),
    Output('neighbor-importance-view', 'figure'),
    Output('cpa-iv-output', 'children'), Output('attention-view', 'children'),
    Output('subgraph-legend-view', 'children'),
    Input('current-predictions-store', 'data'), Input('selected-node-store', 'data'),
    Input('explanation-store', 'data'), Input('cpa-iv-store', 'data'),
    State('precomputed-package-store', 'data'), State('full-dataset-store', 'data'),
    State('editable-graph-store', 'data'), State('model-dropdown', 'value')
)
def update_all_visuals(predictions_data, selected_node, explanation, cpa_data, package, dataset, graph_data, model_type):
    if not all([predictions_data, package, dataset, graph_data]):
        return [], go.Figure(), go.Figure(), "N/A", "N/A", None

    preds = np.array(predictions_data['preds'])
    true_labels = np.array(dataset['y'])
    num_classes = dataset['num_classes']
    color_palette = px.colors.qualitative.Plotly
    class_to_color_map = {i: color_palette[i % len(color_palette)] for i in range(num_classes)}
    
    selected_node_id_str = selected_node['id'] if selected_node else None
    
    embeddings_fig = plot_embeddings(np.array(package['embeddings_2d']['umap']), preds, true_labels, selected_node_id_str, class_to_color_map, num_classes)
    
    graph_elements, neighbor_fig, cpa_panel, attention_panel, legend_content = [], go.Figure(layout={'title': "Select a node to see explanations"}), "Click button to run.", "N/A", None

    if selected_node_id_str and (0 <= int(selected_node_id_str) < dataset['num_nodes']):
        node_idx = int(selected_node_id_str)
        pyg_data = Data(edge_index=torch.tensor(graph_data['edge_index']), num_nodes=dataset['num_nodes'])
        
        # *** FIX: Corrected function call to match new definition ***
        graph_elements, nodes_in_view = create_subgraph_cytoscape(pyg_data, node_idx, preds, class_to_color_map, explanation, cpa_data)
        
        neighbor_fig = plot_2hop_importance(pyg_data, explanation, node_idx)
        if cpa_data: cpa_panel = html.Ul([html.Li(f"Path: {' → '.join(map(str, p['path']))} (Score: {p['score']:.4f})") for p in cpa_data]) if cpa_data else "No paths found."
        
        # *** CHANGE: Add selected node info to the top of the legend ***
        selected_node_info = html.Div([
            html.Strong(f"Selected Node: {node_idx}"), 
            html.P(f"Predicted: {int(preds[node_idx])}", style={'margin': '2px'}), 
            html.P(f"True: {int(true_labels[node_idx])}", style={'margin': '2px'}),
            html.Hr()
        ])
        
        one_hop_nodes, _, _, _ = k_hop_subgraph(node_idx, 1, pyg_data.edge_index, relabel_nodes=False, num_nodes=pyg_data.num_nodes)
        one_hop_set = set(one_hop_nodes.cpu().numpy())
        
        one_hop_items = [html.Strong("1-Hop Neighbors:")]
        two_hop_items = [html.Hr(), html.Strong("2-Hop Neighbors:")]

        # Separate nodes into 1-hop and 2-hop lists
        for n_id in sorted(list(nodes_in_view)):
            if n_id == node_idx: continue
            
            pred_l = int(preds[n_id])
            true_l = int(true_labels[n_id])
            
            item = html.P(f"Node {n_id} -> P: {pred_l}, T: {true_l}", style={'margin': '2px', 'fontSize': '12px'})
            
            if n_id in one_hop_set:
                one_hop_items.append(item)
            else:
                two_hop_items.append(item)

        legend_content = html.Div([selected_node_info] + one_hop_items + two_hop_items)


        if model_type == 'GAT' and package.get('attention_weights'):
            att_data = package['attention_weights']
            att_edge_index, att_scores = att_data['conv2']
            att_scores = torch.tensor(att_scores).mean(dim=1)
            mask = torch.tensor(att_edge_index)[1] == node_idx
            if torch.any(mask):
                att_df = pd.DataFrame({'Neighbor': torch.tensor(att_edge_index)[0][mask].tolist(), 'Attention': att_scores[mask].tolist()}).sort_values('Attention', ascending=False)
                # *** FIX: Add style to GAT attention plot ***
                attention_panel = dcc.Graph(
                    figure=px.bar(att_df.head(15), x='Neighbor', y='Attention', title=f"Top Attention to Node {node_idx}").update_layout(xaxis_type='category', paper_bgcolor='rgba(0,0,0,0)'),
                    style={'height': '280px'}
                )
            else: attention_panel = html.P("No incoming attention.")
        else:
            attention_panel = html.P("Not a GAT model.")

    return graph_elements, embeddings_fig, neighbor_fig, cpa_panel, attention_panel, legend_content

# --- Run Application ---
if __name__ == '__main__':
    print("Dash app starting...")
    app.run(debug=True, host='0.0.0.0', port=8050)
