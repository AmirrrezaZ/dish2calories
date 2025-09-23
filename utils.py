
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


def scale_nutrition_features(
    df: pd.DataFrame,
    scalers: Optional[Dict[str, StandardScaler]] = None,
    cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    """
    Fit StandardScaler on the train split (tag=='train') and apply to all splits.

    Args:
        df: DataFrame with a 'tag' column in {'train','val','test'}.
        scalers: dict of fitted MinMaxScaler per feature, or None to fit on train.
        cols: features to scale (default: ['total_mass','total_fat','total_carb','total_protein']).

    Returns:
        (df_scaled, scalers) where df_scaled has scaled features and scalers
        is the dict of fitted StandardScaler per column.
    """
    if cols is None:
        cols = ['total_mass', 'total_fat', 'total_carb', 'total_protein']
        
    df = df.copy()

    # --- Fit scalers if not provided ---
    if scalers is None:
        df_tr = df.loc[df['tag'] == 'train']
        if df_tr.empty:
            raise ValueError("No rows with tag=='train' found for fitting scalers.")

        scalers = {}
        for col in cols:
            scaler = StandardScaler()
            scaler.fit(df_tr[[col]].to_numpy(dtype=np.float32))
            scalers[col] = scaler

    # --- Transform all splits ---
    for col in cols:
        if col not in scalers:
            raise KeyError(f"Scaler for column '{col}' not found in provided scalers.")
        df.loc[:, col] = scalers[col].transform(df[[col]].to_numpy(dtype=np.float32))

    return df, scalers






def get_all_image_paths(base_dir: str, dish_id: str):
    """
    Collect all image file paths under subfolders of base_dir/images/dish_id.
    Returns sorted list of full paths.
    """
    base_path = Path(base_dir) / "images" / str(dish_id)
    if not base_path.exists():
        raise FileNotFoundError(f"{base_path} does not exist")

    subfolders = sorted([p for p in base_path.iterdir() if p.is_dir()])
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    image_paths = []
    for folder in subfolders:
        image_paths.extend(
            str(p) for p in folder.iterdir() if p.suffix.lower() in image_extensions
        )

    return sorted(image_paths)


def build_metadata(base_dir: str,
                   id_col: str = "id",
                   n: int = -1,
                   seed: int | None = None) -> pd.DataFrame:
    """
    For each id in metadata[id_col], find image paths under base_dir/images/<id>/*,
    select up to n images, and return a DataFrame with all original metadata cols
    plus 'img_path'.
    """
    metadata_path = f"{base_dir}/metadata.csv"
    df = pd.read_csv(metadata_path, index_col=0)

    rng = np.random.default_rng(seed) if seed is not None else None
    rows = []

    for _, row in df.iterrows():
        dish_id = str(row[id_col])
        try:
            paths = get_all_image_paths(base_dir, dish_id)
        except FileNotFoundError:
            continue

        if n == -1 or n >= len(paths):
            selected = paths
        else:
            if rng is None:
                idx = np.random.choice(len(paths), size=n, replace=False)
            else:
                idx = rng.choice(len(paths), size=n, replace=False)
            selected = [paths[i] for i in sorted(idx)]

        for p in selected:
            row_dict = row.to_dict()
            row_dict["img_path"] = p
            rows.append(row_dict)

    return pd.DataFrame(rows)




def stratified_split(df: pd.DataFrame,
                     label_col: str = "label",
                     test_frac: float = 0.2,
                     val_frac: float = 0.1,
                     seed: int = 42) -> pd.DataFrame:
    """
    Split a DataFrame into train/val/test sets with stratification,
    then tag each row with a 'tag' column.

    Args:
        df: Input DataFrame.
        label_col: Column name used for stratification.
        test_frac: Fraction of the data for the test set.
        val_frac: Fraction of the *remaining* train set to use as validation.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with all original columns + a 'tag' column in {"train","val","test"}.
    """
    # --- Split into trainval vs test ---
    trainval, test = train_test_split(
        df,
        test_size=test_frac,
        shuffle=True,
        random_state=seed
    )

    # --- Split trainval into train vs val ---
    train, val = train_test_split(
        trainval,
        test_size=val_frac,
        shuffle=True,
        random_state=seed
    )

    # --- Add tags ---
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["tag"] = "train"
    val["tag"] = "val"
    test["tag"] = "test"

    # --- Combine back ---
    df_tagged = pd.concat([train, val, test], ignore_index=True)
    return df_tagged



def build_classes(df: pd.DataFrame, label_col: str = "label"):
    """
    Build ingredient classes from a DataFrame where labels are stored as strings of lists.
    
    Example row: "['eggs', 'chicken']"
    """
    ingredients = []
    for label in df[label_col].dropna().unique():
        try:
            parsed = ast.literal_eval(label)  # convert string -> list
            ingredients.extend(parsed)
        except Exception:
            continue  # skip malformed rows

    ingredients = np.unique(ingredients)
    cls2idx = {c: i for i, c in enumerate(ingredients)}
    idx2cls = {i: c for i, c in enumerate(ingredients)}
    return ingredients, cls2idx, idx2cls


