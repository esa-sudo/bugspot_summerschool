import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import streamlit as st
import yaml

# Allow importing from ../bugspot-main/src without requiring package install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUGSPOT_SRC = PROJECT_ROOT / "bugspot-main" / "src"
import sys

if str(BUGSPOT_SRC) not in sys.path:
    sys.path.insert(0, str(BUGSPOT_SRC))

from bugspot import DetectionPipeline, get_default_config


PARAM_GROUPS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Background Model (GMM)": {
        "gmm_history": {
            "label": "Background memory length",
            "kind": "int",
            "min": 10,
            "max": 5000,
            "step": 10,
            "help": "How many past image frames are used to learn what counts as the static background. Higher means more stability, but adaptation to real scene changes becomes slower.",
        },
        "gmm_var_threshold": {
            "label": "Motion sensitivity threshold",
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
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Minimum ratio of largest color blob to total motion area in detection region.",
        },
        "max_num_blobs": {
            "label": "Maximum separate motion islands",
            "kind": "int",
            "min": 1,
            "max": 50,
            "step": 1,
            "help": "Upper limit on disconnected moving regions inside one candidate area. Lower values are stricter and reduce clutter-induced detections.",
        },
        "min_motion_ratio": {
            "label": "Minimum motion fill",
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
            "kind": "int",
            "min": 1,
            "max": 100000,
            "step": 1,
            "help": "Smallest allowed detection area in pixels squared. Raise this to ignore dust-like or sensor-noise artifacts.",
        },
        "max_area": {
            "label": "Largest object size",
            "kind": "int",
            "min": 10,
            "max": 500000,
            "step": 10,
            "help": "Largest allowed detection area in pixels squared. Lower this to reject birds, branches, or large shadows.",
        },
        "min_density": {
            "label": "Compactness requirement",
            "kind": "float",
            "min": 0.0,
            "max": 20.0,
            "step": 0.1,
            "help": "How compact the contour should be, based on area relative to boundary length. Higher values reject very stringy or elongated motion patterns.",
        },
        "min_solidity": {
            "label": "Smoothness requirement",
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
            "kind": "float",
            "min": 0.0,
            "max": 1000.0,
            "step": 1.0,
            "help": "Minimum net distance a track must travel to be accepted as real movement. Increase this to suppress stationary false positives.",
        },
        "min_path_points": {
            "label": "Minimum observations per track",
            "kind": "int",
            "min": 1,
            "max": 200,
            "step": 1,
            "help": "Minimum number of time points required before movement-pattern tests are trusted.",
        },
        "max_frame_jump": {
            "label": "Max allowed jump per frame",
            "kind": "float",
            "min": 1.0,
            "max": 1000.0,
            "step": 1.0,
            "help": "Largest position jump allowed between two consecutive image frames for the same track. Lower values prevent accidental links between different objects.",
        },
        "max_lost_frames": {
            "label": "How long to keep missing tracks",
            "kind": "int",
            "min": 1,
            "max": 500,
            "step": 1,
            "help": "Number of frames a track can disappear before deletion. Higher values help reconnect after occlusion; too high can fuse unrelated trajectories.",
        },
        "max_area_change_ratio": {
            "label": "Max size change between frames",
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
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "How much the tracker relies on location when linking detections between frames. Higher values prefer the physically closest match.",
        },
        "tracker_w_area": {
            "label": "Match weight: size",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Importance of size similarity during matching. Higher values favor consistent apparent object size over time.",
        },
        "tracker_cost_threshold": {
            "label": "Strictness of matching",
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
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Maximum fraction of the path that revisits earlier locations. Lower values require more exploratory movement.",
        },
        "min_progression_ratio": {
            "label": "Min forward progression",
            "kind": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Minimum proportion of movement that advances rather than oscillates locally. Higher values require more directed motion.",
        },
        "max_directional_variance": {
            "label": "Max direction variability",
            "kind": "float",
            "min": 0.0,
            "max": 5.0,
            "step": 0.01,
            "help": "How much direction changes are tolerated along a track. Lower values favor steadier heading; higher values accept erratic turns.",
        },
        "revisit_radius": {
            "label": "Distance for counting a revisit",
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


def _render_parameter_inputs(base_config: Dict[str, Any]) -> Dict[str, Any]:
    edited = dict(base_config)

    for group_name, params in PARAM_GROUPS.items():
        with st.expander(group_name, expanded=False):
            for key, meta in params.items():
                current = edited[key]
                widget_key = f"param_{key}"
                if meta["kind"] == "int":
                    edited[key] = st.slider(
                        meta["label"],
                        min_value=int(meta["min"]),
                        max_value=int(meta["max"]),
                        value=int(current),
                        step=int(meta["step"]),
                        help=meta.get("help"),
                        key=widget_key,
                    )
                else:
                    edited[key] = st.slider(
                        meta["label"],
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]),
                        value=float(current),
                        step=float(meta["step"]),
                        help=meta.get("help"),
                        key=widget_key,
                    )

    return _normalize_config(edited)


def _save_uploaded_video(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="bugspot_upload_"))
    video_path = temp_dir / f"input{suffix}"
    video_path.write_bytes(uploaded.getbuffer())
    return video_path


def _annotate_output_video(
    input_video_path: Path,
    output_dir: Path,
    all_detections,
    confirmed_track_ids,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "annotated_result.mp4"

    by_frame: Dict[int, list] = {}
    for det in all_detections:
        by_frame.setdefault(det["frame_number"], []).append(det)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video for annotation")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 25.0

    writer = None
    for codec in ("avc1", "H264", "mp4v"):
        candidate = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        cap.release()
        raise RuntimeError("Could not create a video writer with available codecs")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_dets = by_frame.get(frame_idx, [])
        for det in frame_dets:
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
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Annotated video was not written correctly")

    return out_path


def _run_pipeline(uploaded_video, config: Dict[str, Any]) -> Tuple[Path, Dict[str, Any], str]:
    input_path = _save_uploaded_video(uploaded_video)
    run_dir = Path(tempfile.mkdtemp(prefix="bugspot_run_"))
    crops_dir = run_dir / "crops"
    composites_dir = run_dir / "composites"

    pipeline = DetectionPipeline(config)
    result = pipeline.process_video(
        str(input_path),
        extract_crops=True,
        render_composites=True,
        save_crops_dir=str(crops_dir),
        save_composites_dir=str(composites_dir),
    )

    confirmed = set(result.confirmed_tracks.keys())
    video_path = _annotate_output_video(input_path, run_dir, result.all_detections, confirmed)

    summary = {
        "total_detections": len(result.all_detections),
        "confirmed_tracks": len(result.confirmed_tracks),
        "video_info": result.video_info,
        "output_dir": str(run_dir),
    }

    config_yaml = yaml.safe_dump(config, sort_keys=False)
    return video_path, summary, config_yaml


def main() -> None:
    st.set_page_config(page_title="BugSpot Parameter Playground", layout="wide")
    _inject_tooltip_styles()

    st.title("BugSpot Parameter Playground")
    st.write(
        "Upload a video, tune detection parameters, update the config, run BugSpot, "
        "and preview the annotated result video."
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

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv", "m4v"],
        accept_multiple_files=False,
    )

    st.subheader("Config Parameters")
    edited_config = _render_parameter_inputs(st.session_state.active_config)

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if st.button("Update Config", use_container_width=True):
            st.session_state.active_config = edited_config
            st.session_state.config_yaml = yaml.safe_dump(edited_config, sort_keys=False)
            st.success("Config updated. You can run BugSpot or download this config.")
    with col_b:
        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state.active_config = dict(default_cfg)
            st.session_state.config_yaml = yaml.safe_dump(default_cfg, sort_keys=False)
            st.rerun()
    with col_c:
        st.download_button(
            "Download Current Config",
            data=st.session_state.config_yaml,
            file_name="detection_config_tuned.yaml",
            mime="application/x-yaml",
            use_container_width=True,
        )

    st.code(st.session_state.config_yaml, language="yaml")

    st.subheader("Run Pipeline")
    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        run_clicked = st.button("Process Video", type="primary", use_container_width=True)
    with run_col2:
        if uploaded_video is None:
            st.info("Upload a video first, then click Process Video.")

    if run_clicked:
        if uploaded_video is None:
            st.error("Please upload a video before processing.")
        else:
            with st.spinner("Running BugSpot pipeline..."):
                video_path, summary, cfg_yaml = _run_pipeline(uploaded_video, st.session_state.active_config)
                st.session_state.result_video_path = str(video_path)
                st.session_state.result_video_bytes = video_path.read_bytes()
                st.session_state.result_summary = summary
                st.session_state.config_yaml = cfg_yaml

    if "result_summary" in st.session_state:
        summary = st.session_state.result_summary
        info = summary["video_info"]

        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric("Detections", summary["total_detections"])
        stat2.metric("Confirmed Tracks", summary["confirmed_tracks"])
        stat3.metric("FPS", f"{info['fps']:.2f}")
        stat4.metric("Duration (s)", f"{info['duration']:.2f}")

        st.caption(f"Artifacts saved in: {summary['output_dir']}")

    if "result_video_bytes" in st.session_state:
        st.subheader("Annotated Result Video")
        _, video_col, _ = st.columns([2, 3, 2])
        with video_col:
            st.video(st.session_state.result_video_bytes, format="video/mp4")


if __name__ == "__main__":
    main()
