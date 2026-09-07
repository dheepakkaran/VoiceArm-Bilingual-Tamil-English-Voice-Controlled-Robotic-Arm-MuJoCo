"""VoiceArm dashboard: issue commands, watch execution, inspect episode history."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src import config, episodes
from src.executor import execute
from src.sim import SimEnv

st.set_page_config(page_title="VoiceArm", layout="wide")


@st.cache_resource
def get_env() -> SimEnv:
    env = SimEnv()
    env.park()
    return env


@st.cache_resource
def speech_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401
        import sounddevice  # noqa: F401
    except ImportError:
        return False
    return True


env = get_env()

st.title("VoiceArm")
st.caption("Code-switched speech grounding for open-vocabulary robotic manipulation")

# --- metrics ----------------------------------------------------------------
s = episodes.summary()
cols = st.columns(6)
cols[0].metric("Episodes", s["episodes"])
cols[1].metric("Success rate", f"{s['success_rate'] * 100:.0f}%")
cols[2].metric("Mean detect error", f"{s['mean_detect_error_cm']:.1f} cm")
cols[3].metric("Mean duration", f"{s['mean_duration_s']:.1f} s")
cols[4].metric("Mean LLM latency", f"{s['mean_llm_latency_s']:.2f} s")
cols[5].metric("Tamil ASR won", f"{s['tamil_specialist_share'] * 100:.0f}%")

st.divider()
left, right = st.columns([1, 1])

# --- command input ----------------------------------------------------------
with left:
    st.subheader("Command")
    text = st.text_input("Tamil, English, or Tanglish",
                         placeholder="sivappu block-ah bowl-la vai")

    c1, c2 = st.columns(2)
    run = c1.button("Execute", type="primary", use_container_width=True)

    has_speech = speech_available()
    mic = c2.button("Record 5s", disabled=not has_speech, use_container_width=True)
    if not has_speech:
        st.caption("Speech backend unavailable — install `mlx-whisper` and "
                   "`sounddevice` to enable the microphone. Text input still works.")

    transcript = None
    if mic:
        from src import speech

        with st.spinner("Listening…"):
            tr = speech.route(speech.record(5.0))
        transcript = tr
        text = speech.clean_transcript(tr.text)
        st.write("**ASR candidates**")
        st.dataframe(pd.DataFrame([
            {"model": k, "transcript": v, "chosen": k == tr.source}
            for k, v in tr.candidates.items()
        ]), hide_index=True, use_container_width=True)
        st.write(f"**Cleaned** — {text}")

    if (run or mic) and text:
        with st.spinner("Planning and executing…"):
            ep = execute(env, text, transcript=transcript, capture_video=True)
        st.session_state["last"] = ep

# --- last episode -----------------------------------------------------------
with right:
    st.subheader("Last episode")
    ep = st.session_state.get("last")
    if ep is None:
        st.info("No command run yet.")
    else:
        st.success("Success") if ep.success else st.error("Failed")
        st.write("**Plan**")
        st.json(json.loads(ep.plan_json))
        st.write(f"plan source `{ep.plan_source}` · "
                 f"detection error `{ep.detect_error_m * 100:.1f} cm`" if
                 not np.isnan(ep.detect_error_m) else f"plan source `{ep.plan_source}`")

st.divider()
st.subheader("Scene")
sc1, sc2 = st.columns(2)
sc1.image(env.render(config.SCENE_CAM), caption="scenecam")
sc2.image(env.render(config.TOP_CAM), caption="topcam (perception view)")

# --- history ----------------------------------------------------------------
st.divider()
st.subheader("Episodes")
df = episodes.load_index()
if df.empty:
    st.info("No episodes recorded yet.")
else:
    f1, f2 = st.columns(2)
    langs = f1.multiselect("Language", sorted(df["detected_language"].unique()))
    only_ok = f2.checkbox("Successful only")
    view = df
    if langs:
        view = view[view["detected_language"].isin(langs)]
    if only_ok:
        view = view[view["success"]]
    st.dataframe(
        view[["episode_id", "raw_utterance", "cleaned_utterance", "asr_source",
              "plan_source", "target_object", "detect_error_m", "success",
              "duration_s", "llm_latency_s"]].iloc[::-1],
        hide_index=True, use_container_width=True,
    )
