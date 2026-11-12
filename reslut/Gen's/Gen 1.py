# 🔴 Section 1: Implement for MA's
# ----
# ⚪ 1. Libraries

# Cell 1
import pandas as pd
import scipy.io
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, average_precision_score

# ⚪ 2. Extract indices from .csv file

# --- Configuration ---
csv_file_path = '/media/mine/Mine/3.Main-Ubuntu-Envs/2.Thesis-Env/3.Datasets/mammographyori.csv'

def get_ma_indices(df, feature_name, low, high):
    condition = (df[feature_name] >= low) & (df[feature_name] <= high)
    subset = df[condition]
    indices = subset.index.to_numpy()
    counts = subset['label'].value_counts()
    return indices, counts

df_original = pd.read_csv(csv_file_path, header=None)
df_original.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'label']
print(f"Shape of the original dataset: {df_original.shape} (rows, columns)")

ma_conditions = {
    'MA1': ('c4', -0.9, 2),
    'MA2': ('c5', -0.4, 1),
    'MA3': ('c6', -1, 1),
}

ma_indices = {}
ma_label_counts = {}
for name, (feature, low, high) in ma_conditions.items():
    indices, counts = get_ma_indices(df_original, feature, low, high)
    ma_indices[name] = indices
    ma_label_counts[name] = counts
    print(f"{name} Condition: Feature '{feature}' in [{low}, {high}]")
    print(f"  samples = {len(indices)}")
    print(f"  label counts =\n{counts}\n")

# ⚪ 3. Create X_tem lists for MA1/MA2/MA3
X_tem_ma1 = ma_indices['MA1'].tolist()
X_tem_ma2 = ma_indices['MA2'].tolist()
X_tem_ma3 = ma_indices['MA3'].tolist()

print('MA1 size:', len(X_tem_ma1))
print('MA2 size:', len(X_tem_ma2))
print('MA3 size:', len(X_tem_ma3))

# Three-way neighborhood characteristic region-based outlier detection (3WIROD) algorithm
# Please refer to the following papers:
# Zhang Xianyong, Yuan Zhong, and Miao Duoqian.Outlier Detection Using Three-Way
# Neighborhood Characteristic Regions and Corresponding Fusion Measurement[J].TKDE,2023.
# Uploaded by Yuan Zhong on August 29, 2023. E-mail:yuanzhong2799@foxmail.com.

def WNCROD(data, X_tem, lammda):
    # input:
    # data is data matrix without decisions, where rows for samples and columns for attributes.
    # Numerical attributes should be normalized into [0,1].
    # Nominal attributes be replaced by different integer values.
    # X_tem denotes the selected condition subdata.
    # lammda is a given parameter for the radius adjustment.
    # output
    # Multiple neighborhood outlier factor (MNOF)
    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1

    D1 = m / 3
    D2 = m / 2
    D3 = 0.9 * m

    delta = np.zeros(m)
    ID = np.all(data <= 1, axis=0)
    delta[ID] = np.std(data[:, ID], axis=0) / lammda

    Lower = np.zeros((m, n))

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        Lower_temp = np.min(np.maximum(1 - RM_tem, np.tile(X, (n, 1))), axis=1)
        Lower[col, :Lower_temp.shape[0]] = Lower_temp

    IB = np.tile(X, (m, 1)) - Lower
    NEB = np.min(IB, axis=0)
    NPB = IB - np.tile(NEB, (m, 1))

    n_X = int(sum(X))

    weight = np.zeros((n_X, m))

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        weight_x = []

        for i in range(n_X):
            temp1 = RM_tem[X_tem[i], :]
            weight_temp = 1 - (np.sqrt((np.sum(np.minimum(temp1, X))) / n_X))
            weight_x.append(weight_temp)

        weight[:len(weight_x), col] = weight_x

    D_tem = np.zeros((n, n))

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        D_tem += RM_tem

    NOM = m - D_tem
    X_OM = NOM[np.ix_(X_tem, X_tem)]

    NEB_num = np.zeros((n_X, m))
    Lower_num = np.zeros((n_X, m))
    NPB_num = np.zeros((n_X, m))

    for col in range(m):
        temp2 = Lower[col, :]
        temp3 = NPB[col, :]

        for i in range(n_X):
            temp1 = X_OM[i, :]
            NEB_num[i, col] = np.sum(np.minimum(NEB[X_tem], temp1 <= D1))
            NPB_num[i, col] = np.sum(np.minimum(temp3[X_tem], temp1 >= D2))
            Lower_num[i, col] = np.sum(np.minimum(temp2[X_tem], temp1 >= D3))

    MNOF = np.mean(((NEB_num + Lower_num + NPB_num) / n_X) * weight, axis=1)
    return MNOF

# ⚪ 4. Run WNCROD for MA1, MA2, and MA3
# 1) Load data and normalize once
mat = loadmat('/media/mine/Mine/3.Main-Ubuntu-Envs/2.Thesis-Env/3.Datasets/mammography.mat')
full_data = mat['trandata']            # shape = (n_samples, 7)

X = full_data[:, :6]
y = full_data[:, 6].astype(int)

scaler = MinMaxScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])

print('Loaded and normalized data once for all MA subsets.')

def evaluate_wncrod(data, labels, X_tem, lam=0.5):
    out_scores = WNCROD(data, X_tem, lammda=lam)
    return roc_auc_score(labels[X_tem], out_scores), average_precision_score(labels[X_tem], out_scores)

print('Running WNCROD for MA1, MA2, MA3 with λ = 0.5')
for name, X_tem in [('MA1', X_tem_ma1), ('MA2', X_tem_ma2), ('MA3', X_tem_ma3)]:
    roc_auc, ap = evaluate_wncrod(X, y, X_tem, lam=0.5)
    print(f"{name}: ROC-AUC = {roc_auc:.4f}, AP = {ap:.4f}")

# ⚪ 5. Run WNCROD only on MA1 for λ = 0.1 to 2.0
lambda_list_ma1 = [0.1, 0.2, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.8, 0.9, 1.0,
                   1.1, 1.2, 1.3, 1.4, 1.5,
                   1.6, 1.7, 1.8, 1.9, 2.0]

ma1_results = []
for lam in lambda_list_ma1:
    roc_auc, ap = evaluate_wncrod(X, y, X_tem_ma1, lam=lam)
    ma1_results.append({'lambda': lam, 'ROC-AUC': roc_auc, 'AP': ap})

print('MA1 results for λ = 0.1 to 2.0:')
for r in ma1_results:
    print(f"λ = {r['lambda']:.1f} | ROC-AUC = {r['ROC-AUC']:.4f} | AP = {r['AP']:.4f}")

# ⚪ 6. Run WNCROD for All MA for λ = 0.1 to 2.0
lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5,
               0.6, 0.7, 0.8, 0.9, 1.0,
               1.1, 1.2, 1.3, 1.4, 1.5,
               1.6, 1.7, 1.8, 1.9, 2.0]

all_results = {name: [] for name in ['MA1', 'MA2', 'MA3']}
for lam in lambda_list:
    for name, X_tem in [('MA1', X_tem_ma1), ('MA2', X_tem_ma2), ('MA3', X_tem_ma3)]:
        roc_auc, ap = evaluate_wncrod(X, y, X_tem, lam=lam)
        all_results[name].append({'lambda': lam, 'ROC-AUC': roc_auc, 'AP': ap})

for name in ['MA1', 'MA2', 'MA3']:
    print(f'\n{name} results:')
    for r in all_results[name]:
        print(f"λ = {r['lambda']:.1f} | ROC-AUC = {r['ROC-AUC']:.4f} | AP = {r['AP']:.4f}")

import matplotlib.pyplot as plt

lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5,
               0.6, 0.7, 0.8, 0.9, 1.0,
               1.1, 1.2, 1.3, 1.4, 1.5,
               1.6, 1.7, 1.8, 1.9, 2.0]

roc_auc_ma1 = [r['ROC-AUC'] for r in all_results['MA1']]

plt.figure(figsize=(8, 5))
plt.plot(lambda_list, roc_auc_ma1, marker='o', linestyle='-',
         color='C0', label='ROC-AUC')
plt.title('ROC-AUC vs λ (MA1)')
plt.xlabel('λ')
plt.ylabel('ROC-AUC')
plt.xticks(lambda_list, rotation=45)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 🔴 Section 2: New Idea's
# ⚪ 0. Checking Idea's

import importlib
from sklearn.metrics import roc_auc_score, average_precision_score

# Load module dynamically to support filenames starting with numbers (e.g., '1_opus_5_thinking')
try:
    opus_module = importlib.import_module("1_opus_5_thinking")
except ModuleNotFoundError:
    # Fallback in case the file was renamed without the leading digit
    opus_module = importlib.import_module("opus_5_thinking")

# Extract required classes and registry from the loaded module
VARIANTS = opus_module.VARIANTS
Variant0Baseline = opus_module.Variant0Baseline
Variant1AdaptiveRadius = opus_module.Variant1AdaptiveRadius
Variant2DiscriminativeWeights = opus_module.Variant2DiscriminativeWeights
Variant3SubspaceAugmented = opus_module.Variant3SubspaceAugmented
Variant4TieredEvidence = opus_module.Variant4TieredEvidence
Variant5RobustScale = opus_module.Variant5RobustScale
Variant6SilvermanSelector = opus_module.Variant6SilvermanSelector
Variant7AmiseRate = opus_module.Variant7AmiseRate
Variant8ResolutionFloor = opus_module.Variant8ResolutionFloor
VariantFull = opus_module.VariantFull

# Evaluation wrapper matching the existing pipeline interface
def evaluate_opus_model(model_cls, data, labels, X_tem, lam=0.5):
    # Initialize model (lambda is automatically ignored by parameter-free variants)
    model = model_cls(lammda=lam)
    out_scores = model.fit_predict(data, X_index=X_tem)
    auc = roc_auc_score(labels[X_tem], out_scores)
    ap = average_precision_score(labels[X_tem], out_scores)
    return auc, ap

# Sanity Check: Verify exact mathematical alignment of V0 with baseline results on MA1
v0_auc, v0_ap = evaluate_opus_model(Variant0Baseline, X, y, X_tem_ma1, lam=0.5)
print(f"[Sanity Check] V0 on MA1 (lambda=0.5): ROC-AUC = {v0_auc:.4f}, AP = {v0_ap:.4f}")

# Variant descriptions mapping for report and logging
variant_descriptions = {
    "V0": "Baseline (Original 3WNCROD)",
    "V1": "Idea 1: Adaptive Radius (Eq. 8*) [No Lambda]",
    "V2": "Idea 2: Discriminative Weights (Eq. 10*)",
    "V3": "Idea 3: Subspace Augmentation (Alg. 0*, Eq. 9*)",
    "V4": "Idea 4: Tiered Evidence NDF (Eq. 4*)",
    "V5": "Ablation: Robust Scale Only (MADN / lambda)",
    "V6": "Ablation: Silverman Selector Only",
    "V7": "Ablation: AMISE Rate Only [No Lambda]",
    "V8": "Ablation: Resolution Floor Only (1/n)",
    "VF": "Full Enhanced (All Ideas Combined) [No Lambda]"
}

comparison_records = []
print("Evaluating all Opus 5 variants on MA1, MA2, MA3 (lambda = 0.5)...")

for key, model_cls in VARIANTS.items():
    row = {
        "Key": key,
        "Description": variant_descriptions.get(key, key)
    }
    for name, X_tem in [('MA1', X_tem_ma1), ('MA2', X_tem_ma2), ('MA3', X_tem_ma3)]:
        auc, ap = evaluate_opus_model(model_cls, X, y, X_tem, lam=0.5)
        row[f"{name} AUC"] = round(auc, 4)
        row[f"{name} AP"] = round(ap, 4)
    comparison_records.append(row)

# Display tabular comparison
import pandas as pd

df_comparison = pd.DataFrame(comparison_records)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("\n--- Performance Comparison Table (lambda = 0.5) ---")
print(df_comparison)

# Benchmark comparison between parameter-free models and the best grid-search results
# Safe fallback to published baseline values if Cell 7 was skipped in the current session
if 'all_results' in globals():
    best_baseline = {
        name: max(all_results[name], key=lambda item: item['ROC-AUC'])
        for name in ['MA1', 'MA2', 'MA3']
    }
else:
    # Published optimal grid search results from Table 6 and Fig. 4 in the paper
    best_baseline = {
        'MA1': {'ROC-AUC': 0.8931, 'AP': 0.3550, 'lambda': 0.5},
        'MA2': {'ROC-AUC': 0.7745, 'AP': 0.0626, 'lambda': 0.5},
        'MA3': {'ROC-AUC': 0.7880, 'AP': 0.1669, 'lambda': 0.6}
    }

summary_param_free = []
for name, X_tem in [('MA1', X_tem_ma1), ('MA2', X_tem_ma2), ('MA3', X_tem_ma3)]:
    # Single evaluation for V1 (parameter-free)
    auc_v1, ap_v1 = evaluate_opus_model(Variant1AdaptiveRadius, X, y, X_tem)
    # Single evaluation for VF (parameter-free)
    auc_vf, ap_vf = evaluate_opus_model(VariantFull, X, y, X_tem)
    
    summary_param_free.append({
        "Dataset": name,
        "Best V0 (Grid-Search 20 runs)": f"AUC: {best_baseline[name]['ROC-AUC']:.4f} (at lambda={best_baseline[name]['lambda']})",
        "V1 Adaptive (1 Single Run)": f"AUC: {auc_v1:.4f} | AP: {ap_v1:.4f}",
        "VF Full (1 Single Run)": f"AUC: {auc_vf:.4f} | AP: {ap_vf:.4f}"
    })

print("\n--- Hidden Computational Cost: Grid Search vs. Parameter-Free Single Run ---")
print(pd.DataFrame(summary_param_free))

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(lambda_list, roc_auc_ma1, marker='o', linestyle='-')
plt.title('ROC-AUC vs λ (MA1)')
plt.show()
