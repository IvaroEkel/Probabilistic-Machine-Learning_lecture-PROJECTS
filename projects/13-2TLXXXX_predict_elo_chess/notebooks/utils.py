import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import re
import numpy as np
import chess
import chess.pgn
import glob
import os


# function to filter games based on defined constraints
#TODO i should probably filter out players that have played less than X total games
def keep_game(example) -> bool:
    # 1) move_text contains 'eval'
    if "eval" not in example["movetext"]:
        return False

    # 2) num_moves >= 4
    if "4. " not in example.get("movetext"):
        return False

    # 3) time_control == '600+0'
    if example.get("TimeControl", "") != "600+0":
        return False

    if example.get("WhiteElo") == 1500 or example.get("BlackElo") == 1500:
        return False

    return True

def save_dataframe(df: pd.DataFrame, path: str, compression: str = 'snappy') -> None:
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, path, compression=compression)

def load_dataframe(path: str) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()


def load_all_processed_shards(directory: str, pattern: str = "processed_*.parquet") -> pd.DataFrame:
    shard_paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not shard_paths:
        raise FileNotFoundError("No files found.")

    df_list = []
    for i, path in enumerate(shard_paths):
        df = pd.read_parquet(path)
        df.reset_index(drop=True, inplace=True)
        df['shard_id'] = i  # optional: track source shard
        df_list.append(df)

    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    return full_df


def save_and_verify(df: pd.DataFrame, path: str, compression: str = 'snappy') -> pd.DataFrame:
    save_dataframe(df, path, compression=compression)
    df_loaded = load_dataframe(path)
    pd.testing.assert_frame_equal(df, df_loaded, check_dtype=True, check_exact=True)
    return df_loaded


"""if i want to keep the moves_evals column"""
def prepare_and_save(df: pd.DataFrame, path: str, compression: str = 'snappy') -> None:
    # convert nested DataFrame column to list-of-records for Parquet
    df_copy = df.copy()
    df_copy['moves_evals'] = df_copy['moves_evals'].apply(lambda d: d.to_dict('records'))
    table = pa.Table.from_pandas(df_copy, preserve_index=True)
    pq.write_table(table, path, compression=compression)

def load_with_moves(path: str) -> pd.DataFrame:
    table = pq.read_table(path)
    df = table.to_pandas()
    # reconstruct moves_evals back into DataFrame objects
    df['moves_evals'] = df['moves_evals'].apply(lambda recs: pd.DataFrame(recs))
    return df

def flatten_moves(df: pd.DataFrame) -> pd.DataFrame:
    # explode the list-of-records into one row per move
    df_exp = df.copy()
    df_exp['move_list'] = df_exp['moves_evals'].apply(lambda d: d.to_dict('records'))
    df_exp = df_exp.explode('move_list')
    moves_df = pd.json_normalize(df_exp['move_list'])
    df_flat = pd.concat([df_exp.drop(columns=['moves_evals','move_list']), moves_df], axis=1)
    return df_flat



def parse_moves_and_evals_with_delta(move_text: str) -> pd.DataFrame:
    """
    Parse a Lichess move_text string and extract every move (both White and Black)
    along with its evaluation and the change in evaluation from the previous ply.
    Returns a DataFrame with columns:
      - move_number (int)
      - color ('white' or 'black')
      - move (str, SAN)
      - eval (float or str for mate scores)
      - delta_eval (float, difference from previous eval; NaN for first ply)
    """

    # Patterns for capturing White and Black moves with their evaluations
    white_pattern = re.compile(r'\b(\d+)\.\s*([^\s\{]+)\s*\{\s*\[%eval\s*([^\]\s]+)')
    black_pattern = re.compile(r'\b(\d+)\.\.\.\s*([^\s\{]+)\s*\{\s*\[%eval\s*([^\]\s]+)')

    records = []

    # Extract all White moves
    for w_match in white_pattern.finditer(move_text):
        move_num = int(w_match.group(1))
        move_san = w_match.group(2)
        eval_str = w_match.group(3)
        try:
            eval_val = float(eval_str)
        except ValueError:
            if "-" in eval_str:
                eval_val = -1000
            else:
               eval_val = 1000
        records.append({
            'move_number': move_num,
            'color': 'white',
            'move': move_san,
            'eval': eval_val
        })

    # Extract all Black moves
    for b_match in black_pattern.finditer(move_text):
        move_num = int(b_match.group(1))
        move_san = b_match.group(2)
        eval_str = b_match.group(3)
        try:
            eval_val = float(eval_str)
        except ValueError:
            if "-" in eval_str:
               eval_val = -1000
            else:
               eval_val = 1000
        records.append({
            'move_number': move_num,
            'color': 'black',
            'move': move_san,
            'eval': eval_val
        })

    # Build and sort DataFrame: ensure White before Black
    df_moves = pd.DataFrame(records)
    df_moves['sort_key'] = df_moves['move_number'] * 2 + df_moves['color'].map({'white': 0, 'black': 1})
    df_moves = df_moves.sort_values('sort_key').drop(columns='sort_key').reset_index(drop=True)

    # Compute delta_eval: difference between current eval and previous ply eval
    df_moves['delta_eval'] = df_moves['eval'].diff()
    df_moves.loc[df_moves.index[0], 'delta_eval'] = 0.0

    return df_moves

def _diffs_list(evals_cp, pov_color, mate_threshold=900):
    sign = 1 if pov_color == 'white' else -1
    diffs = []
    for s1, s2 in zip(evals_cp[0::2], evals_cp[1::2]):
        if s1 is None or s2 is None:
            continue
        # skip “mate” as it skews data to much
        # TODO include missed mate as feature
        if abs(s1) >= mate_threshold or abs(s2) >= mate_threshold:
            continue
        d = (s2 - s1) * sign
        diffs.append(max(0, d))
    return diffs


def _accuracy_cp(evals_cp, pov_color):
    diffs = _diffs_list(evals_cp, pov_color)
    if not diffs:
        return None
    return int(round(np.mean(diffs)))

def compute_accuracy_cp_from_moves_evals2(moves_evals):
    """
    Given moves_evals as a list of dicts:
      [{'move_number':1,'color':'white','move':'e4','eval':0.18,'delta_eval':0.00}, … ]
    returns a tuple (white_accuracy_cp, black_accuracy_cp).
    """
    # turn into DataFrame
    g = pd.DataFrame(moves_evals)
    # sort by move_number, with white before black
    g['color_ord'] = (g['color'] == 'black').astype(int)
    g = g.sort_values(['move_number','color_ord'])
    # convert eval (in pawns) → centipawns ints, None if missing
    evals_cp = [
        int(round(v * 100)) if pd.notna(v) else None
        for v in g['eval']
    ]
    w_acc = _accuracy_cp(evals_cp, 'white')
    b_acc = _accuracy_cp(evals_cp, 'black')
    return pd.Series({
        'white_accuracy_cp': w_acc,
        'black_accuracy_cp': b_acc
    })


def compute_accuracy_cp_from_moves_evals(moves_evals):
    """
    Given moves_evals as a list of dicts like
      [{'move_number':1,'color':'white','move':'e4','eval':0.18,'delta_eval':0.00}, … ]
    returns:
      white_accuracy_cp = mean(|delta_eval|*100) for white moves
      black_accuracy_cp = mean(|delta_eval|*100) for black moves
    """
    g = pd.DataFrame(moves_evals)

    # ensure we have delta_eval, drop any missing
    g = g.dropna(subset=['delta_eval'])

    # compute centipawn loss for each move
    g['cp_loss'] = (g['delta_eval'].abs() * 100).astype(int)

    # group by color and average
    white_avg = g.loc[g['color'] == 'white', 'cp_loss'].mean()
    black_avg = g.loc[g['color'] == 'black', 'cp_loss'].mean()

    return pd.Series({
        'white_average_cpl': white_avg,
        'black_average_cpl': black_avg
    })



def clean_san(san: str) -> str:
    """
    Strip move annotations (?, !, +, #) from SAN so python‐chess can parse it.
    """
    return re.sub(r"[?!+#]+$", "", san)


def count_pawn_structure(board: chess.Board, color: bool):
    """
    Count isolated, doubled, and tripled pawns for the given color on 'board'.
    Returns a tuple (isolated_count, doubled_count, tripled_count).
    """
    pawns = list(board.pieces(chess.PAWN, color))
    files = {}
    for sq in pawns:
        f = chess.square_file(sq)
        files.setdefault(f, 0)
        files[f] += 1

    isolated = 0
    doubled = 0
    tripled = 0
    for f, cnt in files.items():
        if cnt == 1:
            # No neighbor pawn on f–1 or f+1
            if (f - 1 not in files) and (f + 1 not in files):
                isolated += 1
        if cnt == 2:
            doubled += 1
        if cnt >= 3:
            tripled += 1
    return isolated, doubled, tripled


def count_center_control(board: chess.Board, color: bool):
    """
    Count how many minor pieces (knights or bishops) of 'color'
    attack or defend any of the four center squares {d4,d5,e4,e5}
    in the given board position.
    """
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    total = 0
    for sq in center_squares:
        # attackers of sq by 'color'
        for attacker in board.attackers(color, sq):
            p = board.piece_at(attacker)
            if p.piece_type in (chess.KNIGHT, chess.BISHOP):
                total += 1
        # defenders are just attackers of sq by the opposite color
        for defender in board.attackers(not color, sq):
            p = board.piece_at(defender)
            if p.piece_type in (chess.KNIGHT, chess.BISHOP):
                total += 1
    return total


def safe_parse_san(board, san):
    """
    Try parsing san, strip annotations and try again,
    return a Move or None if both fail.
    """
    try:
        return board.parse_san(san)
    except chess.IllegalMoveError:
        san2 = re.sub(r"[?!+#]+$", "", san)
        try:
            return board.parse_san(san2)
        except chess.IllegalMoveError:
            return None


def extract_game_features(moves_df: pd.DataFrame) -> dict:
    """
    Given a DataFrame 'moves_df' with columns:
        ["move_number", "color", "move", "eval", "delta_eval"],
    compute features for White and Black, returning a nested dict.

    Replaces any None values with 0 so downstream regressors see only numeric entries.
    """
    illegal = False
    # 1) Determine total full moves
    san_list = []
    for _, row in moves_df.iterrows():
        san = clean_san(row["move"])
        san_list.append(san)
    total_half_moves = len(san_list)
    total_full_moves = (total_half_moves + 1) // 2

    def half_idx_for_frac(f: float) -> int:
        fm = int(f * total_full_moves)
        hi = fm * 2
        return min(hi, total_half_moves)

    quartile_indices = {
        "25": half_idx_for_frac(0.25),
        "50": half_idx_for_frac(0.50),
        "75": half_idx_for_frac(0.75),
        "100": total_half_moves
    }

    # Initialize board and trackers
    board = chess.Board()
    castle_done = {"white": False, "black": False}
    moves_before_castle = {"white": total_full_moves, "black": total_full_moves}

    development_times = {
        "white": {"knights": [], "bishops": [], "rq": []},
        "black": {"knights": [], "bishops": [], "rq": []}
    }

    pawn_counts = {
        "white": {q: (0, 0, 0) for q in quartile_indices},
        "black": {q: (0, 0, 0) for q in quartile_indices}
    }

    legal_diff = {q: 0 for q in quartile_indices}

    center_control = {"white": 0, "black": 0}

    initial_map = {
        sq: (board.piece_at(sq).color, board.piece_at(sq).piece_type, sq)
        for sq in board.piece_map()
    }
    current_to_initial = {sq: info for sq, info in initial_map.items()}
    moved_pieces_after10 = {"white": set(), "black": set()}

    blunder_moves = {"white": [], "black": []}
    mistake_moves = {"white": [], "black": []}

    win_op = {"white": None, "black": None}
    opp_win_op = {"white": None, "black": None}

    knight_edge_first = {"white": None, "black": None}
    knight_edge_count = {"white": 0, "black": 0}
    rook_7th_first = {"white": None, "black": None}

    # Iterate through half-moves
    for i, row in moves_df.iterrows():
        half_idx = i + 1
        color = row["color"]
        move_no = row["move_number"]
        san = clean_san(row["move"])
        d_eval = row["delta_eval"]
        e = row["eval"]

        # Count blunders and mistakes
        if color == "white":
            if d_eval is not None and d_eval < -2.0:
                blunder_moves["white"].append(move_no)
            if d_eval is not None and d_eval < -0.5:
                mistake_moves["white"].append(move_no)
            if win_op["white"] is None and e is not None and e >= 3.0:
                win_op["white"] = move_no
            if opp_win_op["white"] is None and e is not None and e <= -3.0:
                opp_win_op["white"] = move_no
        else:
            if d_eval is not None and d_eval > 2.0:
                blunder_moves["black"].append(move_no)
            if d_eval is not None and d_eval > 0.5:
                mistake_moves["black"].append(move_no)
            if win_op["black"] is None and e is not None and e <= -3.0:
                win_op["black"] = move_no
            if opp_win_op["black"] is None and e is not None and e >= 3.0:
                opp_win_op["black"] = move_no

        mv = safe_parse_san(board, san)
        if mv is None:
            # mark this game illegal and stop processing
            illegal = True
            break

        if illegal:
            # return minimal dict that flags the problem
            return {"illegal_game": True}

        # Moves before castling
        if not castle_done[color] and san.startswith("O-O"):
            castle_done[color] = True
            moves_before_castle[color] = move_no

        # Development
        piece = board.piece_at(mv.from_square)
        if piece:
            cstr = "white" if piece.color == chess.WHITE else "black"
            ptype = piece.piece_type
            if ptype == chess.KNIGHT:
                development_times[cstr]["knights"].append(move_no)
            if ptype == chess.BISHOP:
                development_times[cstr]["bishops"].append(move_no)
            if ptype in (chess.ROOK, chess.QUEEN):
                development_times[cstr]["rq"].append(move_no)

        # Knight on edge & rook to 7th
        if piece and piece.piece_type == chess.KNIGHT:
            to_file = chess.square_file(mv.to_square)
            if to_file in (0, 7):
                if knight_edge_first[color] is None:
                    knight_edge_first[color] = move_no
                knight_edge_count[color] += 1

        if piece and piece.piece_type == chess.ROOK:
            to_rank = chess.square_rank(mv.to_square)
            if color == "white" and to_rank == 6 and rook_7th_first["white"] is None:
                rook_7th_first["white"] = move_no
            if color == "black" and to_rank == 1 and rook_7th_first["black"] is None:
                rook_7th_first["black"] = move_no

        # Unique pieces moved after 10 full moves
        if piece:
            info = current_to_initial.get(mv.from_square)
            if info:
                cstr = "white" if piece.color == chess.WHITE else "black"
                moved_pieces_after10[cstr].add(info[2])

        board.push(mv)

        # Pawn structure & legal-move diff at quartiles
        for label, idx in quartile_indices.items():
            if half_idx == idx:
                w_iso, w_dbl, w_tri = count_pawn_structure(board, chess.WHITE)
                b_iso, b_dbl, b_tri = count_pawn_structure(board, chess.BLACK)
                pawn_counts["white"][label] = (w_iso, w_dbl, w_tri)
                pawn_counts["black"][label] = (b_iso, b_dbl, b_tri)

                turn_saved = board.turn
                board.turn = chess.WHITE
                w_moves = board.legal_moves.count()
                board.turn = chess.BLACK
                b_moves = board.legal_moves.count()
                board.turn = turn_saved
                legal_diff[label] = w_moves - b_moves

    # Development before queen/rooks
    dev_before = {"white": 0, "black": 0}
    for color in ("white", "black"):
        knights = development_times[color]["knights"]
        bishops = development_times[color]["bishops"]
        rq = development_times[color]["rq"]
        if knights and bishops and rq:
            if max(knights) < min(rq) and max(bishops) < min(rq):
                dev_before[color] = 1

    # Center control after 5 full moves (half-move idx = 10)
    board5 = chess.Board()
    for i, row in moves_df.iterrows():
        if i == 10:
            break
        san = clean_san(row["move"])
        mv = safe_parse_san(board5, clean_san(row.move))
        if mv is None:
            # This really shouldn’t happen if the game was legal above,
            # but guard just in case:
            return {"illegal_game": True}
        board5.push(mv)
    center_control["white"] = count_center_control(board5, chess.WHITE)
    center_control["black"] = count_center_control(board5, chess.BLACK)

    # Unique pieces moved after 10 full moves (half-move idx = 20)
    board10 = chess.Board()
    moved_after10 = {"white": set(), "black": set()}
    for i, row in moves_df.iterrows():
        mv = safe_parse_san(board10, clean_san(row.move))
        if mv is None:
            # This really shouldn’t happen if the game was legal above,
            # but guard just in case:
            return {"illegal_game": True}
        piece = board10.piece_at(mv.from_square)
        if piece:
            cstr = "white" if piece.color == chess.WHITE else "black"
            info = {sq: info for sq, info in current_to_initial.items() if info[0] == piece.color}
            initial_sq = None
            for sq, inf in info.items():
                if inf[1] == piece.piece_type and sq == mv.from_square:
                    initial_sq = inf[2]
                    break
            if initial_sq:
                moved_after10[cstr].add(initial_sq)
        board10.push(mv)
        if i == 19:
            break

    unique_after10 = {
        "white": len(moved_after10["white"]),
        "black": len(moved_after10["black"])
    }

    # First blunder/mistake and counts
    first_blunder = {
        "white": blunder_moves["white"][0] if blunder_moves["white"] else None,
        "black": blunder_moves["black"][0] if blunder_moves["black"] else None
    }
    blunder_counts = {
        "white": len(blunder_moves["white"]),
        "black": len(blunder_moves["black"])
    }
    first_mistake = {
        "white": mistake_moves["white"][0] if mistake_moves["white"] else None,
        "black": mistake_moves["black"][0] if mistake_moves["black"] else None
    }
    mistake_counts = {
        "white": len(mistake_moves["white"]),
        "black": len(mistake_moves["black"])
    }

    # Moves before castling for colors that never castled
    for c in ("white", "black"):
        if not castle_done[c]:
            moves_before_castle[c] = total_full_moves

    # Replace all None with 0 in these nested dicts
    for d in (first_blunder, first_mistake, win_op, opp_win_op, knight_edge_first, rook_7th_first):
        for color in ("white", "black"):
            if d[color] is None:
                d[color] = 0


    # Gather everything into a single dict
    features = {
        "total_full_moves": total_full_moves,
        "moves_before_castle": moves_before_castle,
        "pawn_counts_at": pawn_counts,               # nested by color → quartile → (iso,dbl,tri)
        "legal_move_diff_at": legal_diff,            # by quartile → (White legal − Black legal)
        "development_before": dev_before,            # 1 or 0
        "center_control_after_5": center_control,    # minor‐piece control counts
        "unique_pieces_after_10": unique_after10,    # count of distinct initial squares moved
        "blunder_counts": blunder_counts,            # total # of blunders
        "first_blunder_move": first_blunder,         # full‐move index
        "mistake_counts": mistake_counts,            # total # of mistakes
        "first_mistake_move": first_mistake,         # full‐move index
        "first_win_opportunity": win_op,             # full‐move for win opportunity
        "first_opp_win_opportunity": opp_win_op,     # full‐move for opponent win opp
        "knight_edge_first": knight_edge_first,      # full‐move of first knight on edge
        "knight_edge_count": knight_edge_count,      # total times knight moved to edge
        "rook_7th_first": rook_7th_first,          # full‐move of first rook→7th/2nd
        "illegal_game": False
    }

    return features


def add_aggregated_player_and_opening_features(df):
    # 4. Aggregations per player
    # White stats
    white_stats = df.groupby('White').agg(
        w_total_games=('white_win', 'count'),
        w_win_rate=('white_win', 'mean'),
        w_avg_accuracy=('accuracy_cp', 'mean')
    )
    # Black stats
    black_stats = df.groupby('Black').agg(
        b_total_games=('black_win', 'count'),
        b_win_rate=('black_win', 'mean'),
        b_avg_accuracy=('accuracy_cp', 'mean')
    )
    # Merge back
    df = df.merge(white_stats, left_on='White', right_index=True, how='left')
    df = df.merge(black_stats, left_on='Black', right_index=True, how='left')

    # 5. Difference in player aggregated stats
    df['diff_total_games'] = df['w_total_games'] - df['b_total_games']
    df['diff_win_rate'] = df['w_win_rate'] - df['b_win_rate']
    df['diff_avg_accuracy'] = df['w_avg_accuracy'] - df['b_avg_accuracy']

    # 6. Rolling performance: last 10 games win rate
    df['w_roll_win10'] = df.groupby('White')['white_win']\
                            .apply(lambda x: x.shift().rolling(10, min_periods=1).mean())
    df['b_roll_win10'] = df.groupby('Black')['black_win']\
                            .apply(lambda x: x.shift().rolling(10, min_periods=1).mean())
    df['roll_win_diff10'] = df['w_roll_win10'] - df['b_roll_win10']

    # 7. Opening (ECO) stats
    eco_stats = df.groupby('ECO').agg(
        eco_games=('ECO', 'count'),
        eco_avg_elo=('elo_diff', lambda x: ((x + df.loc[x.index, 'BlackElo']*2)/2).mean()),
        eco_white_win_rate=('white_win', 'mean')
    )
    df = df.merge(eco_stats, on='ECO', how='left')

    return df

import pandas as pd


def join_parquet_dataframes(
    file_path_1: str,
    file_path_2: str,
    subset: list[str] | None = None,
    how: str = 'outer',
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Load two Parquet files into pandas DataFrames, perform a union-style join (concatenation),
    and remove duplicate entries.

    Parameters
    ----------
    file_path_1 : str
        Path to the first Parquet file.
    file_path_2 : str
        Path to the second Parquet file.
    subset : list[str], optional
        List of column names to consider when identifying duplicates. If None,
        duplicates are detected across all columns.
    how : {'outer', 'inner', 'left', 'right'}, default 'outer'
        Type of join to perform when combining the two DataFrames:
        - 'outer': include all rows from both DataFrames
        - 'inner': include only rows with matching index values
        - 'left'/'right': include all rows from one DataFrame and matching rows from the other
    keep : {'first', 'last', False}, default 'first'
        Which duplicates (if any) to keep:
        - 'first': keep the first occurrence
        - 'last': keep the last occurrence
        - False: drop all duplicates

    Returns
    -------
    pd.DataFrame
        The combined DataFrame with duplicates removed.

    Examples
    --------
    """
    # Load the two DataFrames
    df1 = pd.read_parquet(file_path_1)
    df2 = pd.read_parquet(file_path_2)

    # Perform the join (concatenation)
    if how == 'outer':
        combined = pd.concat([df1, df2], ignore_index=True)
    else:
        # Align on index; reset indices to merge
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        combined = df1.merge(df2, how=how)

    # Drop duplicates
    combined_deduped = combined.drop_duplicates(subset=subset, keep=keep)

    return combined_deduped


def aggregate_ingame_metrics(
    df: pd.DataFrame,
    group_col: str = 'ECO',
    metrics: list[str] | None = None,
    agg_funcs: list[str] = ['mean', 'median', 'std'],
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Aggregate in-game metrics by a grouping column (e.g., opening or player) and merge back,
    replacing any NaNs in the aggregates with `fill_value`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing game-level features including in-game metrics.
    group_col : str, default 'ECO'
        Column name to group by (e.g., 'ECO', 'Opening').
    metrics : list[str], optional
        List of metric columns to aggregate. If None, defaults to common in-game stats.
    agg_funcs : list[str], default ['mean','median','std']
        Aggregation functions to apply.
    fill_value : float, default 0.0
        Value to substitute for NaNs in the aggregated results.

    Returns
    -------
    pd.DataFrame
        DataFrame with new aggregated feature columns (no NaNs) merged on `group_col`.
    """
    if metrics is None:
        metrics = [
           'white_accuracy_cp', 'black_accuracy_cp',
           'first_mistake_move_white', 'first_mistake_move_black',
           'first_blunder_move_white', 'first_blunder_move_black',
           'first_opp_win_opportunity_white', 'first_opp_win_opportunity_black',
           'blunder_counts_white', 'blunder_counts_black',
           'total_full_moves'
        ]

    available = [m for m in metrics if m in df.columns]
    if not available:
        raise ValueError("None of the specified metrics are present in the DataFrame.")

    # 1) Group & aggregate
    agg_df = (
        df.groupby(group_col)[available]
          .agg(agg_funcs)
    )

    # 2) Flatten the MultiIndex
    agg_df.columns = [f"{group_col}_{col}_{func}"
                      for col, func in agg_df.columns]
    agg_df = agg_df.reset_index()

    # 3) Replace NaNs in the aggregates
    #    This covers both groups that had no data and std-of-one=value cases.
    agg_df = agg_df.fillna(fill_value)

    # 4) Merge back to original
    merged = df.merge(agg_df, on=group_col, how='left')
    return merged


from typing import Optional, List, Tuple, Dict, Any, Union
import pandas as pd
import numpy as np


def remove_collinear_features(
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        target: Optional[Union[str, pd.Series]] = None,
        corr_threshold: float = 0.9,
        desired_features: Optional[int] = None,
        corr_method: str = "pearson",
) -> Dict[str, Any]:
    """
    Iteratively remove collinear features from df until either:
      - no pair of features has abs(correlation) >= corr_threshold, OR
      - number of remaining features <= desired_features (if provided)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : Optional[List[str]]
        List of column names to consider for elimination. If None, all numeric columns are used.
    target : Optional[str or pd.Series]
        If provided, used to decide which feature to drop among a highly correlated pair:
        the feature with LOWER absolute correlation with the target is dropped.
    corr_threshold : float
        Absolute correlation threshold above which two features are considered collinear.
    desired_features : Optional[int]
        If provided, algorithm will stop when number of features <= desired_features.
    corr_method : str
        Correlation method to use (e.g., "pearson", "spearman").

    Returns
    -------
    dict with keys:
      - 'reduced_df' : pd.DataFrame containing only the retained features
      - 'selected_features' : List[str] kept feature names
      - 'removed_features' : List[Tuple[str, str, float, str]] log of removals:
            (dropped_feature, kept_feature, correlation_value, reason)
    """
    # Validate inputs
    if features is None:
        # select numeric columns only
        working_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Requested features not in DataFrame: {missing}")
        working_cols = list(features)

    if target is not None:
        # if target is a column name, extract series; if Series provided, use directly
        if isinstance(target, str):
            if target not in df.columns:
                raise ValueError("target column not found in DataFrame")
            target_series = df[target]
        elif isinstance(target, pd.Series):
            target_series = target
        else:
            raise ValueError("target must be a column name or a pandas Series")
        if not np.issubdtype(target_series.dtype, np.number):
            raise ValueError("target must be numeric for correlation-based decision")

    # defensive copy of working cols
    cols = working_cols.copy()
    removed: List[Tuple[str, str, float, str]] = []

    # Helper to compute absolute correlation matrix for current cols
    def abs_corr_matrix(columns: List[str]) -> pd.DataFrame:
        if len(columns) == 0:
            return pd.DataFrame()
        return df[columns].corr(method=corr_method).abs()

    # Iteratively remove features
    while True:
        # Stop condition if desired_features reached
        if desired_features is not None and len(cols) <= desired_features:
            break

        if len(cols) <= 1:
            break

        corr_mat = abs_corr_matrix(cols)
        # zero diagonal to ignore self-correlation
        np.fill_diagonal(corr_mat.values, 0.0)

        # find max correlation value and corresponding pair
        max_corr_val = corr_mat.values.max()
        if np.isnan(max_corr_val):
            break

        if max_corr_val < corr_threshold:
            # no pair above threshold: done
            break

        # find a pair with that max correlation
        # corr_mat.stack() returns a Series indexed by (i,j)
        stacked = corr_mat.stack()
        pair = stacked.idxmax()  # tuple (feature_i, feature_j)
        feat_a, feat_b = pair
        corr_value = stacked.max()

        # Decide which feature to drop
        reason = ""
        drop = None
        keep = None

        if target is not None:
            # compute abs corr with target
            corr_a = abs(df[feat_a].corr(target_series))
            corr_b = abs(df[feat_b].corr(target_series))
            if np.isnan(corr_a): corr_a = 0.0
            if np.isnan(corr_b): corr_b = 0.0
            if corr_a > corr_b:
                drop, keep = feat_b, feat_a
                reason = f"lower_corr_with_target ({corr_b:.4f} < {corr_a:.4f})"
            elif corr_b > corr_a:
                drop, keep = feat_a, feat_b
                reason = f"lower_corr_with_target ({corr_a:.4f} < {corr_b:.4f})"
            else:
                # tie -> fall back to mean-correlation heuristic below
                pass

        if drop is None:
            # compute mean absolute correlation of each feature to others
            mean_corr = corr_mat.mean(axis=0)  # mean abs corr with other cols (diagonal already zero)
            # drop the feature with higher mean correlation (more "collinear")
            if mean_corr[feat_a] >= mean_corr[feat_b]:
                drop, keep = feat_a, feat_b
                reason = f"higher_mean_corr ({mean_corr[feat_a]:.4f} >= {mean_corr[feat_b]:.4f})"
            else:
                drop, keep = feat_b, feat_a
                reason = f"higher_mean_corr ({mean_corr[feat_b]:.4f} > {mean_corr[feat_a]:.4f})"

        # perform drop
        cols.remove(drop)
        removed.append((drop, keep, float(corr_value), reason))

    reduced_df = df[cols].copy()
    return {
        "reduced_df": reduced_df,
        "selected_features": cols,
        "removed_features": removed,
    }

