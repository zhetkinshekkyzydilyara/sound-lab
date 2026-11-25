"""
Дыбыс толқындары мен жиілікті цифрлық зерттеуге арналған шағын қосымша.
Іске қосу:  streamlit run app.py
"""
import io
import wave

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

SR = 44100          # дискреттеу жиілігі, Гц
V_SOUND = 343.0     # ауадағы дыбыс жылдамдығы (20 °С), м/с

st.set_page_config(page_title="Дыбыс зертханасы", page_icon="🔊", layout="wide")


# ---------- Көмекші функциялар ----------
def make_wave(kind, f, a, t):
    """Берілген пішіндегі толқын: y(t)"""
    phase = 2 * np.pi * f * t
    if kind == "Синусоида":
        return a * np.sin(phase)
    if kind == "Тікбұрышты":
        return a * np.sign(np.sin(phase))
    if kind == "Үшбұрышты":
        return a * (2 / np.pi) * np.arcsin(np.sin(phase))
    if kind == "Ара тәрізді":
        return a * 2 * (f * t - np.floor(0.5 + f * t))
    return np.zeros_like(t)


def to_wav(y):
    """numpy массивін WAV байттарына айналдыру (тыңдау үшін)"""
    y = np.clip(y, -1, 1)
    data = (y * 32767 * 0.8).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    return buf.getvalue()


def spectrum(y):
    """Жылдам Фурье түрлендіруі (FFT): жиіліктер мен амплитудалар"""
    n = len(y)
    amp = np.abs(np.fft.rfft(y)) * 2 / n
    freqs = np.fft.rfftfreq(n, 1 / SR)
    return freqs, amp


# ---------- Басқару панелі ----------
qp = st.query_params
modes = ["Бір толқын", "Екі толқын (соғу)"]
kinds = ["Синусоида", "Тікбұрышты", "Үшбұрышты", "Ара тәрізді"]

st.sidebar.header("⚙️ Параметрлер")
mode = st.sidebar.radio("Режим", modes, index=modes.index(qp.get("mode", modes[0])))
kind = st.sidebar.selectbox("Толқын пішіні", kinds, index=kinds.index(qp.get("kind", kinds[0])))
f1 = st.sidebar.slider("Жиілік f₁, Гц", 20, 2000, int(qp.get("f", 440)), step=1)
a1 = st.sidebar.slider("Амплитуда A", 0.1, 1.0, float(qp.get("a", 1.0)), step=0.1)
if mode == modes[1]:
    f2 = st.sidebar.slider("Жиілік f₂, Гц", 20, 2000, int(qp.get("f2", 444)), step=1)
window_ms = st.sidebar.slider("Графиктегі уақыт аралығы, мс", 2, 1000,
                              int(qp.get("win", 10)), step=1)
dur = 2.0  # дыбыс ұзақтығы, с

# ---------- Есептеу ----------
t = np.arange(int(SR * dur)) / SR
y = make_wave(kind, f1, a1, t)
if mode == modes[1]:
    y = (y + make_wave(kind, f2, a1, t)) / 2

T = 1 / f1
lam = V_SOUND / f1
omega = 2 * np.pi * f1

# ---------- Негізгі бет ----------
st.title("🔊 Дыбыс толқындары мен жиілікті цифрлық зерттеу")
st.caption("Толқынның параметрлерін өзгертіп, графиктерді бақылаңыз және дыбысты тыңдаңыз.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Жиілік f", f"{f1} Гц")
c2.metric("Период T = 1/f", f"{T * 1000:.3f} мс")
c3.metric("Циклдік жиілік ω = 2πf", f"{omega:.0f} рад/с")
c4.metric("Толқын ұзындығы λ = v/f", f"{lam:.3f} м")
if mode == modes[1]:
    fb = abs(f1 - f2)
    d1, d2, d3 = st.columns(3)
    d1.metric("Соғу жиілігі |f₁ − f₂|", f"{fb} Гц")
    d2.metric("Соғу периоды", f"{1 / fb:.3f} с" if fb else "—")
    d3.metric("Орташа жиілік", f"{(f1 + f2) / 2:.1f} Гц")

left, right = st.columns(2)

with left:
    st.subheader("📈 Толқын графигі y(t)")
    n = int(SR * window_ms / 1000)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(t[:n] * 1000, y[:n], color="#1f77b4", lw=1.6)
    ax.set_xlabel("Уақыт t, мс")
    ax.set_ylabel("Ауытқу y")
    ax.set_ylim(-1.15, 1.15)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with right:
    st.subheader("📊 Жиілік спектрі (FFT)")
    freqs, amp = spectrum(y)
    fmax = max(2000, 6 * max(f1, f2 if mode == modes[1] else 0))
    m = freqs <= fmax
    if mode == modes[1]:            # соғу режимінде екі шыңды жақыннан көрсету
        lo, hi = min(f1, f2) - 15, max(f1, f2) + 15
        m = (freqs >= lo) & (freqs <= hi)
    fig2, ax2 = plt.subplots(figsize=(7, 3.6))
    ax2.plot(freqs[m], amp[m], color="#d62728", lw=1.2)
    ax2.set_xlabel("Жиілік, Гц")
    ax2.set_ylabel("Амплитуда")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

st.subheader("🎧 Дыбысты тыңдау")
st.audio(to_wav(y), format="audio/wav")

# Спектрдегі ең үлкен 5 шың (гармоникалар)
st.subheader("🔍 Спектрдегі негізгі шыңдар")
idx = np.argsort(amp)[::-1]
peaks = []
for i in idx:
    if amp[i] < 0.01:
        break
    if all(abs(freqs[i] - p[0]) > 1.5 for p in peaks):
        peaks.append((freqs[i], amp[i]))
    if len(peaks) == 5:
        break
peaks.sort()
st.table({
    "Жиілік, Гц": [f"{p[0]:.0f}" for p in peaks],
    "Амплитуда": [f"{p[1]:.3f}" for p in peaks],
    "f / f₁ қатынасы": [f"{p[0] / f1:.2f}" for p in peaks],
})

with st.expander("📘 Формулалар"):
    st.latex(r"y(t) = A\sin(2\pi f t),\quad T=\frac{1}{f},\quad \omega = 2\pi f,\quad \lambda=\frac{v}{f}")
    st.latex(r"f_{\text{соғу}} = |f_1 - f_2|")
    st.write("Ауадағы дыбыс жылдамдығы v ≈ 343 м/с (20 °С). Адам құлағы 20 Гц – 20 000 Гц аралығын естиді.")
