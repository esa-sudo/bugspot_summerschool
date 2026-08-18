import bisect
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st
import yaml

# Allow importing from ../bugspot-main/src without requiring package install.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[0] if APP_DIR.name != "" else APP_DIR
BUGSPOT_SRC = APP_DIR / "bugspot-main" / "src"

if str(BUGSPOT_SRC) not in sys.path:
    sys.path.insert(0, str(BUGSPOT_SRC))

from bugspot import DetectionPipeline, get_default_config

TEST_VIDEOS_DIR = APP_DIR / "test_videos"
CACHE_DIR = APP_DIR / ".cache"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


PARAM_GROUPS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Background Model (GMM)": {
        "gmm_history": {
            "label": "Background memory length",
            "unit": "frames",
            "kind": "int",
            "min": 10,
            "max": 5000,
            "step": 10,
            "help": "How many past image frames are used to learn what counts as the static background. Higher means more stability, but adaptation to real scene changes becomes slower.",
        },
        "gmm_var_threshold": {
            "label": "Motion sensitivity threshold",
            "unit": "intensity²",
            "kind": "float",
            "min": 1.0,
            "max": 100.0,
            "step": 0.5,
            "help": "Controls how different a pixel must be from background before it is treated as motion. Lower values detect subtle movement but can add noise; higher values are stricter and cleaner but may miss faint insect motion.",
        },
    },
    "Morphological Filtering": {
        "morph_kernel_size": {
            "label": "Noise cleanup strength",
            "unit": "px",
            "kind": "int",
            "min": 1,
            "max": 31,
            "step": 2,
            "help": "Cleans up after motion extraction, this filters noise based on its size and connectivity. Bigger values remove speckle and leaf shimmer better, but can erase very small insects.",
        },
    },
    "Cohesiveness": {
        "min_largest_blob_ratio": {
            "label": "Require one dominant motion blob",
            "unit": "ratio",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Minimum ratio of largest color blob to total motion area in detection region.",
        },
        "max_num_blobs": {
            "label": "Maximum separate motion islands",
            "unit": "blobs",
            "kind": "int",
            "min": 1,
            "max": 50,
            "step": 1,
            "help": "Upper limit on disconnected moving regions inside one candidate area. Lower values are stricter and reduce clutter-induced detections.",
        },
        "min_motion_ratio": {
            "label": "Minimum motion fill",
            "unit": "ratio",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Minimum fraction of the candidate box that must truly be moving. Higher values require denser, more convincing motion; lower values accept sparse movement.",
        },
    },
    "Shape": {
        "min_area": {
            "label": "Smallest object size",
            "unit": "px²",
            "kind": "int",
            "min": 1,
            "max": 100000,
            "step": 1,
            "help": "Smallest allowed detection area in pixels squared. Raise this to ignore dust-like or sensor-noise artifacts.",
        },
        "max_area": {
            "label": "Largest object size",
            "unit": "px²",
            "kind": "int",
            "min": 10,
            "max": 500000,
            "step": 10,
            "help": "Largest allowed detection area in pixels squared. Lower this to reject birds, branches, or large shadows.",
        },
        "min_density": {
            "label": "Compactness requirement",
            "unit": "px",
            "kind": "float",
            "min": 0.0,
            "max": 20.0,
            "step": 0.1,
            "help": "How compact the contour should be, based on area relative to boundary length. Higher values reject very stringy or elongated motion patterns.",
        },
        "min_solidity": {
            "label": "Smoothness requirement",
            "unit": "ratio",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "How fully the contour fills its own convex envelope. Higher values prefer smoother and less ragged shapes.",
        },
    },
    "Tracking": {
        "min_displacement": {
            "label": "Minimum net travel to confirm",
            "unit": "px",
            "kind": "float",
            "min": 0.0,
            "max": 1000.0,
            "step": 1.0,
            "help": "Minimum net distance a track must travel to be accepted as real movement. Increase this to suppress stationary false positives.",
        },
        "min_path_points": {
            "label": "Minimum observations per track",
            "unit": "frames",
            "kind": "int",
            "min": 1,
            "max": 200,
            "step": 1,
            "help": "Minimum number of time points required before movement-pattern tests are trusted.",
        },
        "max_frame_jump": {
            "label": "Max allowed jump per frame",
            "unit": "px/frame",
            "kind": "float",
            "min": 1.0,
            "max": 1000.0,
            "step": 1.0,
            "help": "Largest position jump allowed between two consecutive image frames for the same track. Lower values prevent accidental links between different objects.",
        },
        "max_lost_frames": {
            "label": "How long to keep missing tracks",
            "unit": "frames",
            "kind": "int",
            "min": 1,
            "max": 500,
            "step": 1,
            "help": "Number of frames a track can disappear before deletion. Higher values help reconnect after occlusion; too high can fuse unrelated trajectories.",
        },
        "max_area_change_ratio": {
            "label": "Max size change between frames",
            "unit": "×",
            "kind": "float",
            "min": 1.0,
            "max": 20.0,
            "step": 0.1,
            "help": "Largest multiplicative size change allowed from one frame to the next for a consistent track.",
        },
    },
    "Tracker Matching": {
        "tracker_w_dist": {
            "label": "Match weight: position",
            "unit": "weight",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "How much the tracker relies on location when linking detections between frames. Higher values prefer the physically closest match.",
        },
        "tracker_w_area": {
            "label": "Match weight: size",
            "unit": "weight",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Importance of size similarity during matching. Higher values favor consistent apparent object size over time.",
        },
        "tracker_cost_threshold": {
            "label": "Strictness of matching",
            "unit": "ratio",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Maximum acceptable mismatch score when linking detections to existing tracks. Lower values are stricter and start new tracks more often.",
        },
    },
    "Path Topology": {
        "max_revisit_ratio": {
            "label": "Max repeated-position behavior",
            "unit": "ratio",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Maximum fraction of the path that revisits earlier locations. Lower values require more exploratory movement.",
        },
        "min_progression_ratio": {
            "label": "Min forward progression",
            "unit": "ratio",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Minimum proportion of movement that advances rather than oscillates locally. Higher values require more directed motion.",
        },
        "max_directional_variance": {
            "label": "Max direction variability",
            "unit": "rad",
            "kind": "float",
            "min": 0.0,
            "max": 5.0,
            "step": 0.01,
            "help": "How much direction changes are tolerated along a track. Lower values favor steadier heading; higher values accept erratic turns.",
        },
        "revisit_radius": {
            "label": "Distance for counting a revisit",
            "unit": "px",
            "kind": "float",
            "min": 1.0,
            "max": 2000.0,
            "step": 1.0,
            "help": "Spatial radius in pixels used to decide if a point is considered a return to a previously visited area.",
        },
    },
}


def _inject_tooltip_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-baseweb="tooltip"] {
            max-width: 560px !important;
            padding: 0.85rem 1rem !important;
            font-size: 1rem !important;
            line-height: 1.55 !important;
        }

        button[aria-label="Show help"] {
            transform: scale(1.15);
            margin-left: 0.25rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    # OpenCV morphology kernel should be positive odd.
    kernel = int(normalized.get("morph_kernel_size", 3))
    if kernel < 1:
        kernel = 1
    if kernel % 2 == 0:
        kernel += 1
    normalized["morph_kernel_size"] = kernel
    return normalized


def _param_label(meta: Dict[str, Any]) -> str:
    unit = meta.get("unit")
    return f"{meta['label']} ({unit})" if unit else meta["label"]


def _render_parameter_inputs(base_config: Dict[str, Any]) -> Dict[str, Any]:
    edited = dict(base_config)

    for group_name, params in PARAM_GROUPS.items():
        with st.expander(group_name, expanded=False):
            for key, meta in params.items():
                current = edited[key]
                widget_key = f"param_{key}"
                label = _param_label(meta)
                if meta["kind"] == "int":
                    edited[key] = st.slider(
                        label,
                        min_value=int(meta["min"]),
                        max_value=int(meta["max"]),
                        value=int(current),
                        step=int(meta["step"]),
                        help=meta.get("help"),
                        key=widget_key,
                    )
                else:
                    edited[key] = st.slider(
                        label,
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]),
                        value=float(current),
                        step=float(meta["step"]),
                        help=meta.get("help"),
                        key=widget_key,
                    )

    return _normalize_config(edited)


def _list_test_videos() -> List[Path]:
    if not TEST_VIDEOS_DIR.exists():
        return []
    return sorted(
        p for p in TEST_VIDEOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _save_uploaded_videos(uploaded_files) -> None:
    TEST_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        dest = TEST_VIDEOS_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())


def _config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _cache_paths(config_hash: str, video_stem: str) -> Tuple[Path, Path, Path]:
    result_dir = CACHE_DIR / config_hash / video_stem
    return result_dir, result_dir / "annotated_result.mp4", result_dir / "summary.json"


def _load_cached_result(config_hash: str, video_stem: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    _, video_path, summary_path = _cache_paths(config_hash, video_stem)
    if video_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        return video_path.read_bytes(), summary
    return None


def _store_cached_result(
    config_hash: str, video_stem: str, video_bytes: bytes, summary: Dict[str, Any]
) -> None:
    result_dir, video_path, summary_path = _cache_paths(config_hash, video_stem)
    result_dir.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(video_bytes)
    summary_path.write_text(json.dumps(summary))


def _build_track_trails(
    all_detections,
) -> Tuple[Dict[str, List[int]], Dict[str, List[Tuple[int, int]]], Dict[str, Tuple[int, int]]]:
    """Per track: sorted frame numbers, matching centroid points, and (first, last) active frame."""
    track_points: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for det in all_detections:
        track_id = det.get("track_id") or "unknown"
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        track_points[track_id].append((det["frame_number"], cx, cy))

    frame_numbers: Dict[str, List[int]] = {}
    trail_points: Dict[str, List[Tuple[int, int]]] = {}
    frame_range: Dict[str, Tuple[int, int]] = {}
    for track_id, pts in track_points.items():
        pts.sort(key=lambda p: p[0])
        frame_numbers[track_id] = [p[0] for p in pts]
        trail_points[track_id] = [(p[1], p[2]) for p in pts]
        frame_range[track_id] = (pts[0][0], pts[-1][0])

    return frame_numbers, trail_points, frame_range


def _annotate_output_video(
    input_video_path: Path,
    output_dir: Path,
    all_detections,
    confirmed_track_ids,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "annotated_result.mp4"

    by_frame: Dict[int, list] = defaultdict(list)
    for det in all_detections:
        by_frame[det["frame_number"]].append(det)

    track_frame_numbers, track_trail_points, track_frame_range = _build_track_trails(all_detections)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video for annotation")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Video has no readable frames")
    height, width = frame.shape[:2]

    # Encode with ffmpeg (H.264/yuv420p) instead of cv2.VideoWriter so the
    # result plays back in-browser via st.video — OpenCV's own writer codecs
    # (mp4v, etc.) are not reliably playable in HTML5 <video>.
    ffmpeg_cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_idx = 0
    while ok:
        # Track trail: draw the path travelled so far, only while this track
        # is active (between its first and last detection frame).
        for track_id, (first_frame, last_frame) in track_frame_range.items():
            if not (first_frame <= frame_idx <= last_frame):
                continue
            idx = bisect.bisect_right(track_frame_numbers[track_id], frame_idx)
            if idx < 2:
                continue
            confirmed = track_id in confirmed_track_ids
            color = (0, 220, 0) if confirmed else (0, 165, 255)
            trail = np.array(track_trail_points[track_id][:idx], dtype=np.int32)
            cv2.polylines(frame, [trail], isClosed=False, color=color, thickness=3)

        for det in by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            track_id = det.get("track_id") or "unknown"
            short_id = str(track_id)[:8]
            confirmed = track_id in confirmed_track_ids
            color = (0, 220, 0) if confirmed else (0, 165, 255)
            label = f"{short_id} {'CONF' if confirmed else 'CAND'}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        cv2.putText(
            frame,
            f"frame {frame_idx}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        proc.stdin.write(frame.tobytes())
        frame_idx += 1
        ok, frame = cap.read()

    cap.release()
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Annotated video was not written correctly (ffmpeg encoding failed)")

    return out_path


def _process_video(video_path: Path, config: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    run_dir = Path(tempfile.mkdtemp(prefix="bugspot_run_"))
    try:
        pipeline = DetectionPipeline(config)
        result = pipeline.process_video(
            str(video_path),
            extract_crops=False,
            render_composites=False,
        )

        confirmed = set(result.confirmed_tracks.keys())
        out_path = _annotate_output_video(video_path, run_dir, result.all_detections, confirmed)

        summary = {
            "total_detections": len(result.all_detections),
            "confirmed_tracks": len(result.confirmed_tracks),
            "video_info": result.video_info,
        }
        return out_path.read_bytes(), summary
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _get_or_process(
    video_path: Path, config: Dict[str, Any], config_hash: str
) -> Tuple[bytes, Dict[str, Any]]:
    cached = _load_cached_result(config_hash, video_path.stem)
    if cached is not None:
        return cached
    video_bytes, summary = _process_video(video_path, config)
    _store_cached_result(config_hash, video_path.stem, video_bytes, summary)
    return video_bytes, summary


def _display_result(video_name: str, video_bytes: bytes, summary: Dict[str, Any], key_prefix: str) -> None:
    info = summary["video_info"]

    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Detections", summary["total_detections"])
    stat2.metric("Confirmed Tracks", summary["confirmed_tracks"])
    stat3.metric("FPS", f"{info['fps']:.2f}")
    stat4.metric("Duration (s)", f"{info['duration']:.2f}")

    st.video(video_bytes, format="video/mp4", width=480)
    st.download_button(
        "Download Annotated Video",
        data=video_bytes,
        file_name=f"bugspot_{Path(video_name).stem}_annotated.mp4",
        mime="video/mp4",
        use_container_width=True,
        key=f"download_{key_prefix}_{video_name}",
    )


def _video_nav(videos: List[Path], key_prefix: str) -> Path:
    total = len(videos)
    select_key = f"select_{key_prefix}"

    nav_prev, nav_label, nav_next = st.columns([1, 5, 1])
    with nav_prev:
        if st.button("◀", use_container_width=True, key=f"prev_{key_prefix}"):
            st.session_state.video_idx = (st.session_state.video_idx - 1) % total
            # Keep the selectbox's own remembered value in sync — Streamlit
            # restores a keyed widget's last value on rerun, which would
            # otherwise silently overrule this button click.
            st.session_state[select_key] = st.session_state.video_idx
    with nav_next:
        if st.button("▶", use_container_width=True, key=f"next_{key_prefix}"):
            st.session_state.video_idx = (st.session_state.video_idx + 1) % total
            st.session_state[select_key] = st.session_state.video_idx

    with nav_label:
        selected_idx = st.selectbox(
            "Video",
            options=list(range(total)),
            index=st.session_state.video_idx,
            format_func=lambda i: videos[i].name,
            label_visibility="collapsed",
            key=select_key,
        )
    if selected_idx != st.session_state.video_idx:
        st.session_state.video_idx = selected_idx

    current_video = videos[st.session_state.video_idx]
    st.caption(f"Video {st.session_state.video_idx + 1} of {total}: {current_video.name}")
    return current_video


def _render_single_mode(videos: List[Path], config: Dict[str, Any], config_hash: str) -> None:
    current_video = _video_nav(videos, key_prefix="single")
    cached = _load_cached_result(config_hash, current_video.stem)

    process_col, status_col = st.columns([1, 3])
    with process_col:
        process_clicked = st.button(
            "Process Video", type="primary", use_container_width=True, key="process_single"
        )
    with status_col:
        if cached is not None:
            st.success("Loaded from cache for the current config — no reprocessing needed.")
        else:
            st.info("Not processed yet for the current config.")

    if process_clicked:
        with st.spinner(f"Processing {current_video.name}..."):
            video_bytes, summary = _get_or_process(current_video, config, config_hash)
        _display_result(current_video.name, video_bytes, summary, key_prefix="single")
    elif cached is not None:
        video_bytes, summary = cached
        _display_result(current_video.name, video_bytes, summary, key_prefix="single")


def _render_batch_mode(videos: List[Path], config: Dict[str, Any], config_hash: str) -> None:
    total = len(videos)
    cached_count = sum(1 for v in videos if _load_cached_result(config_hash, v.stem) is not None)
    st.caption(f"{cached_count}/{total} videos already processed for the current config.")

    if st.button("Process All Videos", type="primary", use_container_width=True, key="process_all"):
        progress = st.progress(0.0)
        status = st.empty()
        preview = st.empty()
        for i, video in enumerate(videos):
            status.write(f"Processing {video.name} ({i + 1}/{total})...")
            video_bytes, summary = _get_or_process(video, config, config_hash)
            with preview.container():
                _display_result(video.name, video_bytes, summary, key_prefix="batch_preview")
            progress.progress((i + 1) / total)
        status.success("All videos processed.")

    st.divider()
    st.subheader("Review a processed video")
    current_video = _video_nav(videos, key_prefix="batch")
    cached = _load_cached_result(config_hash, current_video.stem)
    if cached is not None:
        video_bytes, summary = cached
        _display_result(current_video.name, video_bytes, summary, key_prefix="batch_browse")
    else:
        st.info("This video hasn't been processed yet for the current config. Click 'Process All Videos' above.")


def main() -> None:
    st.set_page_config(page_title="BugSpot Parameter Playground", layout="wide")
    _inject_tooltip_styles()

    st.title("BugSpot Parameter Playground")
    st.write(
        "Tune detection parameters on the left, then process the built-in test videos "
        "and preview the annotated results on the right."
    )
    st.caption(
        "Use the help icon next to each setting to see a plain-language explanation "
        "and the trade-off when you move that slider."
    )

    default_cfg = get_default_config()

    if "active_config" not in st.session_state:
        st.session_state.active_config = dict(default_cfg)
    if "config_yaml" not in st.session_state:
        st.session_state.config_yaml = yaml.safe_dump(st.session_state.active_config, sort_keys=False)
    if "video_idx" not in st.session_state:
        st.session_state.video_idx = 0

    videos = _list_test_videos()
    if not videos:
        st.subheader("Upload Test Videos")
        st.info(
            "No videos found on the server yet. Download your test videos from the test_video folder on your drive."
            "Then, upload them here."
            "They only need to be uploaded once per server session."
        )
        uploaded_videos = st.file_uploader(
            "Upload test videos",
            type=["mp4", "mov", "avi", "mkv", "m4v"],
            accept_multiple_files=True,
        )
        if uploaded_videos:
            with st.spinner("Saving uploaded videos..."):
                _save_uploaded_videos(uploaded_videos)
            st.rerun()
        return
    st.session_state.video_idx %= len(videos)

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Config Parameters")
        edited_config = _render_parameter_inputs(st.session_state.active_config)

        if st.button("Update Config", use_container_width=True):
            st.session_state.active_config = edited_config
            st.session_state.config_yaml = yaml.safe_dump(edited_config, sort_keys=False)
            st.success("Config updated. Unchanged videos stay cached; new results use this config.")
        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state.active_config = dict(default_cfg)
            st.session_state.config_yaml = yaml.safe_dump(default_cfg, sort_keys=False)
            st.rerun()
        st.download_button(
            "Download Current Config",
            data=st.session_state.config_yaml,
            file_name="detection_config_tuned.yaml",
            mime="application/x-yaml",
            use_container_width=True,
        )

        with st.expander("Current config (YAML)", expanded=False):
            st.code(st.session_state.config_yaml, language="yaml")

    config_hash = _config_hash(st.session_state.active_config)

    with right_col:
        st.subheader("Test Videos")
        with st.expander("Add more videos", expanded=False):
            more_uploads = st.file_uploader(
                "Upload additional test videos",
                type=["mp4", "mov", "avi", "mkv", "m4v"],
                accept_multiple_files=True,
                key="more_video_uploader",
            )
            if more_uploads and st.button("Save uploaded videos", key="save_more_videos"):
                with st.spinner("Saving uploaded videos..."):
                    _save_uploaded_videos(more_uploads)
                st.rerun()

        tab_single, tab_batch = st.tabs(["Single Video", "Process All"])
        with tab_single:
            _render_single_mode(videos, st.session_state.active_config, config_hash)
        with tab_batch:
            _render_batch_mode(videos, st.session_state.active_config, config_hash)


if __name__ == "__main__":
    main()
