import cv2
import time
import csv
import os
import json
from collections import deque, Counter
from datetime import datetime
from ultralytics import YOLO

# ---------------- CONFIG ----------------
CAM_INDEX = 0
MODEL_PATH = "yolov8n.pt"
CONF_TH = 0.40

SHOW_WINDOW = True
PRESS_Q_TO_QUIT = True
RECORD_VIDEO = True
SAVE_RAW_CSV = True
SAVE_STABLE_CSV = True
SAVE_STABLE_JSONL = True

# Grid only: 4 zones = 2x2
GRID_N = 2
TOP_K = 4

# Sampling / stability (CS-only logic)
OBSERVE_EVERY_SEC = 3.0       # make one observation every 3 seconds
DWELL_TIME_SEC = 30.0         # require ~30 seconds of consistency
STABLE_MAJORITY_RATIO = 0.70  # 70% of recent samples should agree
MIN_CONF_FOR_STABLE = 0.55    # optional confidence floor for accepted stable state

# Rolling metadata export for ISE
EXPORT_EVERY_SEC = 120.0
ISE_OUT_DIR = "ise_exports"

# Saving
OUT_DIR = "runs"
RAW_CSV_NAME = "raw_observations.csv"
STABLE_CSV_NAME = "stable_states.csv"
STABLE_JSONL_NAME = "stable_states.jsonl"
VIDEO_NAME = "realtime_record.mp4"
VIDEO_FOURCC = "mp4v"
# ----------------------------------------


# ---------- Grid helpers ----------
def grid_cell_from_point(x, y, W, H, n=2):
    cell_w = W / n
    cell_h = H / n
    col = int(x // cell_w) + 1
    row = int(y // cell_h) + 1
    col = min(max(col, 1), n)
    row = min(max(row, 1), n)
    return row, col


def grid_index(row, col, n=2):
    return (row - 1) * n + (col - 1)


def draw_grid(frame, W, H, n=2, color=(255, 255, 255), thickness=2):
    for i in range(1, n):
        x = int(W * i / n)
        y = int(H * i / n)
        cv2.line(frame, (x, 0), (x, H), color, thickness)
        cv2.line(frame, (0, y), (W, y), color, thickness)


# ---------- Top-K formatting ----------
def top_k_from_counts(counts, k=4):
    idxs = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    out = []
    for idx in idxs[:k]:
        if counts[idx] <= 0:
            break
        out.append((idx, counts[idx]))
    return out


def format_top_grid(top_list, n=2):
    if not top_list:
        return "None"
    parts = []
    for idx, cnt in top_list:
        row = (idx // n) + 1
        col = (idx % n) + 1
        parts.append(f"({row},{col})={cnt}")
    return " ".join(parts)


# ---------- State logic ----------
def dominant_from_counts(counts):
    max_val = max(counts) if counts else 0
    if max_val <= 0:
        return None
    idx = counts.index(max_val)
    return idx


def make_state_signature(people, dom_grid_idx, valid):
    # Lightweight state for temporal stability tracking
    return (people, dom_grid_idx, valid)


def analyze_stability(history, min_ratio=0.70):
    """
    history: deque of state signatures
    returns: (is_stable, dominant_signature, ratio)
    """
    if not history:
        return False, None, 0.0

    counts = Counter(history)
    dominant_signature, freq = counts.most_common(1)[0]
    ratio = freq / len(history)
    return ratio >= min_ratio, dominant_signature, ratio


def grid_idx_to_label(idx, n=2):
    if idx is None:
        return "None"
    row = (idx // n) + 1
    col = (idx % n) + 1
    return f"G({row},{col})"


def build_packet(ts_iso, people, dom_grid_idx, avg_conf, valid, grid_counts):
    return {
        "timestamp": ts_iso,
        "occupancy_count": people,
        "dominant_grid": grid_idx_to_label(dom_grid_idx, GRID_N),
        "avg_confidence": round(avg_conf, 3),
        "valid": int(valid),
        "grid_counts": grid_counts,
    }


# ---------- Saving ----------
def make_run_paths():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_DIR, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    raw_csv_path = os.path.join(run_dir, RAW_CSV_NAME)
    stable_csv_path = os.path.join(run_dir, STABLE_CSV_NAME)
    stable_jsonl_path = os.path.join(run_dir, STABLE_JSONL_NAME)
    video_path = os.path.join(run_dir, VIDEO_NAME)
    return run_dir, raw_csv_path, stable_csv_path, stable_jsonl_path, video_path


def export_packet_for_ise(packet, out_dir=ISE_OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"ise_metadata_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(packet, f, ensure_ascii=False, indent=2)
    return path


# ---------- Main ----------
def main():
    run_dir, raw_csv_path, stable_csv_path, stable_jsonl_path, video_path = make_run_paths()
    print(f"[INFO] Output folder: {run_dir}")
    print(f"[INFO] ISE export folder: {ISE_OUT_DIR}")

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try CAM_INDEX=0/1 or check permissions.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps is None or cam_fps <= 1 or cam_fps != cam_fps:
        cam_fps = 30.0

    out = None
    if RECORD_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
        out = cv2.VideoWriter(video_path, fourcc, cam_fps, (W, H))
        if not out.isOpened():
            video_path = video_path.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(video_path, fourcc, cam_fps, (W, H))
            if not out.isOpened():
                raise RuntimeError("Cannot open VideoWriter. Try changing codec or permissions.")

    raw_csv_file = None
    raw_writer = None
    if SAVE_RAW_CSV:
        raw_csv_file = open(raw_csv_path, "w", newline="", encoding="utf-8")
        raw_writer = csv.writer(raw_csv_file)
        grid_headers = [f"g{r}{c}" for r in range(1, GRID_N + 1) for c in range(1, GRID_N + 1)]
        raw_writer.writerow([
            "frame", "t_sec", "sample_id",
            "people", "avg_conf", "valid",
            "dominant_grid",
            *grid_headers,
            "top_grid",
            "proc_ms", "fps"
        ])
        raw_csv_file.flush()

    stable_csv_file = None
    stable_writer = None
    if SAVE_STABLE_CSV:
        stable_csv_file = open(stable_csv_path, "w", newline="", encoding="utf-8")
        stable_writer = csv.writer(stable_csv_file)
        stable_writer.writerow([
            "stable_event_id", "timestamp", "people", "avg_conf", "valid",
            "dominant_grid", "stability_ratio"
        ])
        stable_csv_file.flush()

    stable_jsonl_file = None
    if SAVE_STABLE_JSONL:
        stable_jsonl_file = open(stable_jsonl_path, "w", encoding="utf-8")

    frame_id = 0
    sample_id = 0
    stable_event_id = 0
    t0 = time.time()
    last_fps_t = time.time()
    fps_smooth = 0.0

    history_len = max(1, int(round(DWELL_TIME_SEC / OBSERVE_EVERY_SEC)))
    state_history = deque(maxlen=history_len)
    last_observe_t = 0.0
    last_committed_signature = None
    last_committed_packet = None
    last_export_t = 0.0

    print("[INFO] Running CS-only real-time perception.")
    print(f"[INFO] Grid mode: {GRID_N}x{GRID_N} (4 zones only)")
    print(f"[INFO] Observation every {OBSERVE_EVERY_SEC:.1f}s, dwell {DWELL_TIME_SEC:.1f}s, history size={history_len}")
    print(f"[INFO] ISE metadata export every {EXPORT_EVERY_SEC:.0f}s")
    print("[INFO] Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_id += 1
            start = time.perf_counter()
            now = time.time()
            t_sec = now - t0

            results = model.predict(frame, conf=CONF_TH, verbose=False)[0]
            vis = frame.copy()

            draw_grid(vis, W, H, n=GRID_N)

            grid_counts = [0] * (GRID_N * GRID_N)
            confs = []

            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls != 0:  # person only
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confs.append(conf)

                    bx = int((x1 + x2) / 2)
                    by = int(y2)  # bottom-center

                    gr, gc = grid_cell_from_point(bx, by, W, H, GRID_N)
                    gidx = grid_index(gr, gc, GRID_N)
                    grid_counts[gidx] += 1

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.circle(vis, (bx, by), 4, (255, 255, 255), -1)
                    cv2.putText(
                        vis,
                        f"G({gr},{gc}) {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

            people = sum(grid_counts)
            avg_conf = (sum(confs) / len(confs)) if confs else 0.0
            valid = 1 if (people > 0 and avg_conf >= CONF_TH) else 0
            dom_grid_idx = dominant_from_counts(grid_counts)

            top_grid = top_k_from_counts(grid_counts, TOP_K)
            top_grid_text = format_top_grid(top_grid, GRID_N)

            proc_ms = (time.perf_counter() - start) * 1000.0

            dt = now - last_fps_t
            if dt > 0:
                fps_inst = 1.0 / dt
                fps_smooth = fps_inst if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * fps_inst)
            last_fps_t = now

            # Observation sampling for temporal stability
            stable_ratio = 0.0
            stable_signature = None
            is_stable = False
            sampled_this_loop = False

            if (now - last_observe_t) >= OBSERVE_EVERY_SEC:
                sample_id += 1
                sampled_this_loop = True
                current_signature = make_state_signature(people, dom_grid_idx, valid)
                state_history.append(current_signature)
                last_observe_t = now

                is_stable, stable_signature, stable_ratio = analyze_stability(
                    state_history,
                    min_ratio=STABLE_MAJORITY_RATIO,
                )

                if (
                    is_stable
                    and stable_signature is not None
                    and stable_signature != last_committed_signature
                    and avg_conf >= MIN_CONF_FOR_STABLE
                    and stable_signature[2] == 1
                ):
                    stable_people, stable_grid, stable_valid = stable_signature
                    ts_iso = datetime.now().isoformat(timespec="seconds")
                    packet = build_packet(
                        ts_iso,
                        stable_people,
                        stable_grid,
                        avg_conf,
                        stable_valid,
                        grid_counts,
                    )

                    last_committed_signature = stable_signature
                    last_committed_packet = packet
                    stable_event_id += 1

                    print("\n[STABLE STATE COMMITTED]")
                    print(json.dumps(packet, indent=2))

                    if stable_writer is not None:
                        stable_writer.writerow([
                            stable_event_id,
                            ts_iso,
                            stable_people,
                            f"{avg_conf:.3f}",
                            stable_valid,
                            grid_idx_to_label(stable_grid, GRID_N),
                            f"{stable_ratio:.2f}",
                        ])
                        stable_csv_file.flush()

                    if stable_jsonl_file is not None:
                        stable_jsonl_file.write(json.dumps(packet) + "\n")
                        stable_jsonl_file.flush()

                if raw_writer is not None:
                    raw_writer.writerow([
                        frame_id,
                        f"{t_sec:.2f}",
                        sample_id,
                        people,
                        f"{avg_conf:.3f}",
                        valid,
                        grid_idx_to_label(dom_grid_idx, GRID_N),
                        *grid_counts,
                        top_grid_text,
                        f"{proc_ms:.2f}",
                        f"{fps_smooth:.2f}",
                    ])
                    raw_csv_file.flush()
            else:
                is_stable, stable_signature, stable_ratio = analyze_stability(
                    state_history,
                    min_ratio=STABLE_MAJORITY_RATIO,
                )

            # Rolling export to ISE every 2 minutes using latest committed stable state
            if last_committed_packet is not None and (now - last_export_t) >= EXPORT_EVERY_SEC:
                export_path = export_packet_for_ise(last_committed_packet, ISE_OUT_DIR)
                print(f"[ISE EXPORT] {export_path}")
                last_export_t = now

            # ---------- overlays ----------
            cv2.putText(
                vis,
                f"Frame:{frame_id} | People:{people} | Valid:{bool(valid)} | Conf:{avg_conf:.2f} | FPS:{fps_smooth:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"Now: Grid={grid_idx_to_label(dom_grid_idx, GRID_N)} | TopG:{top_grid_text}",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"Observe every {OBSERVE_EVERY_SEC:.0f}s | Dwell {DWELL_TIME_SEC:.0f}s | Stable={is_stable} ({stable_ratio:.2f})",
                (20, 101),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            status_line = "No committed state yet"
            if last_committed_packet is not None:
                status_line = (
                    f"Committed: {last_committed_packet['dominant_grid']} / "
                    f"count={last_committed_packet['occupancy_count']}"
                )
            cv2.putText(
                vis,
                status_line,
                (20, 134),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                f"ISE export every {EXPORT_EVERY_SEC:.0f}s to folder: {ISE_OUT_DIR}",
                (20, 167),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            if sampled_this_loop:
                cv2.putText(
                    vis,
                    f"[Sample taken #{sample_id}]",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (255, 255, 255),
                    2,
                )

            if RECORD_VIDEO and out is not None:
                out.write(vis)

            if SHOW_WINDOW:
                cv2.imshow("CS-Only Real-Time YOLO + Stable Grid", vis)
                if PRESS_Q_TO_QUIT and (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted (Ctrl+C).")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            if out is not None:
                out.release()
        except Exception:
            pass
        try:
            if raw_csv_file is not None:
                raw_csv_file.close()
        except Exception:
            pass
        try:
            if stable_csv_file is not None:
                stable_csv_file.close()
        except Exception:
            pass
        try:
            if stable_jsonl_file is not None:
                stable_jsonl_file.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[INFO] Done.")
        if SAVE_RAW_CSV:
            print(f"[INFO] Raw CSV:      {raw_csv_path}")
        if SAVE_STABLE_CSV:
            print(f"[INFO] Stable CSV:   {stable_csv_path}")
        if SAVE_STABLE_JSONL:
            print(f"[INFO] Stable JSONL: {stable_jsonl_path}")
        if RECORD_VIDEO:
            print(f"[INFO] Video:        {video_path}")
        print(f"[INFO] ISE exports:  {ISE_OUT_DIR}")


if __name__ == "__main__":
    main()
