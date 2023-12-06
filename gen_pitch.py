import argparse
import os
import shutil
import subprocess
import tempfile
import warnings
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import parselmouth
import tqdm

import magic


FRAME_PER_SEC = 15
TONES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', "A", "A#", "B"]


class Tonality:
    base_diff = [0, 2, 4, 5, 7, 9, 11]
    tone_freq_map = {
        f"{t}{i}": 2 ** (TONES.index(t) / 12 + i) * 16.3516
        for t in TONES for i in range(0, 8)
    }
    def __init__(self, tone):
        assert tone in TONES
        self.tone = tone
        self.scale = [TONES[(TONES.index(self.tone) + diff) % len(TONES)] for diff in self.base_diff]

    @classmethod
    def normalize_to_freq(cls, tone_or_freq):
        if isinstance(tone_or_freq, (float, int)):
            return tone_or_freq
        return cls.tone_freq_map[tone_or_freq]

    def get_tone_and_freq(self, min_freq=0, max_freq=4186):
        min_freq = self.normalize_to_freq(min_freq)
        max_freq = self.normalize_to_freq(max_freq)

        ret = []
        for i in range(0, 8):
            for base_tone in self.scale:
                tone = f"{base_tone}{i}"
                freq = self.tone_freq_map[tone]
                if min_freq <= freq <= max_freq:
                    ret.append((tone, freq))

        ret.sort()
        return ret


def draw_standard(tone, min_freq, max_freq):
    labels = []
    for tone, f in Tonality(tone).get_tone_and_freq(min_freq, max_freq):
        plt.axline((0, f), (1, f), lw=2)
        label = plt.text(1.01, f, tone, ha='left', va='bottom', fontsize=18)
        label.set_visible(False)
        labels.append(label)
    return labels


def animate(frame, pitch, ln, mid_ln, progress_bar, labels):
    m = magic.magic()
    progress_bar.update(1)
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    curr_time = frame / FRAME_PER_SEC
    time_start = curr_time - 2.5
    time_end = curr_time + 2.5
    pitch_in_range = [p for p in zip(pitch.xs(), pitch_values) if time_start <= p[0] <= time_end]
    pitch_xs = [p[0] for p in pitch_in_range]
    pitch_vals = [p[1] for p in pitch_in_range]
    assert 0 not in pitch_vals
    ln.set_data(pitch_xs, pitch_vals)
    mid_ln.set_data([[curr_time, curr_time], [0, 1]])

    # Calculate the average pitch only with the middle part of pitch_vals
    # np.nanmean will generate a warning if all values are nan, ignore it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avr_pitch = np.nanmean(pitch_vals[len(pitch_vals) // 3: len(pitch_vals) * 2 // 3])

    if not np.isnan(avr_pitch):
        pitch_low, pitch_high = (avr_pitch * (m[2233] ** (-m[41]/m[1823])), avr_pitch * (m[72] ** (m[1795]/m[1823])))
        getattr(plt, "".join(chr(m[i]) for i in [490, 403, 343, 367]))(pitch_low, pitch_high)

        # Set labels' visibilities based on pitch
        for label in labels:
            _, y = label.get_position()
            if pitch_low <= y <= pitch_high:
                label.set_visible(True)
            else:
                label.set_visible(False)

    for label in labels:
        _, y = label.get_position()
        label.set_position((time_start + 0.01, y))

    getattr(plt, "".join(chr(m[i]) for i in [190, 275, 135, 193]))(time_start, time_end)


def generate_pitch_video(path, output, tone, min_freq, max_freq):
    # Get all the pitch in the audio and plot it
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch_ac(pitch_floor=min_freq, pitch_ceiling=max_freq)
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(19.2, 10.8), layout="tight")

    labels = draw_standard(tone, min_freq, max_freq)
    plt.xlim([snd.xmin, snd.xmax])

    # Draw the mid line to indicate the current time
    ln, = plt.plot([0], [0], '.', markersize=10, color="orange")
    mid_ln = plt.axvline(0, color="red")

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.yscale('log')
    plt.ylabel("fundamental frequency [Hz]")

    render_time = snd.xmax

    print("Generating pitch video")

    with tqdm.tqdm(total=int(render_time * FRAME_PER_SEC), unit="frame", ncols=100) as progress_bar:
        # Do the animation and save to mp4
        ani = animation.FuncAnimation(
            fig,
            partial(animate, pitch=pitch, ln=ln, mid_ln=mid_ln, progress_bar=progress_bar, labels=labels),
            frames=range(int(render_time * FRAME_PER_SEC)))

        FFWriter = animation.FFMpegWriter(fps=FRAME_PER_SEC)
        ani.save(output, writer=FFWriter)

    plt.close()


def combine_video(ffmpeg, video_path, pitch_video_path, output_path, pitch_width, pitch_position):
    print("Combining video")

    if pitch_width is None:
        # Get the resolution of the original video
        process = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=s=x:p=0',
            video_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        width, _ = process.stdout.decode().split("x")

        # Scale the pitch video to half of the original video
        pitch_width = int(width) // 2

    if pitch_position == "top_right":
        overlay_param = f'W-w-10:10'
    elif pitch_position == "top_left":
        overlay_param = f'10:10'
    elif pitch_position == "bottom_right":
        overlay_param = f'W-w-10:H-h-10'
    elif pitch_position == "bottom_left":
        overlay_param = f'10:H-h-10'
    else:
        raise ValueError(f"Invalid pitch position {pitch_position}")

    with magic.magic3() as m:
        print(f"Writing to {os.path.abspath(output_path)}")
        subprocess.run([
            ffmpeg,
            '-loglevel', 'error',
            '-stats',
            '-i', video_path,
            '-i', pitch_video_path,
            *m,
            f'[1:v]scale={pitch_width}:-1 [scaled_ol]; [2:v]scale={pitch_width}:-1 [scaled_wm]; '
            f'[0:v][scaled_ol]overlay={overlay_param}[temp]; [temp][scaled_wm]overlay={overlay_param}[outv]',
            '-map', '[outv]',
            '-map', '0:a',
            output_path], check=True)


if __name__ == "__main__":
    magic.magic()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, help="path to the input audio file", required=True)
    parser.add_argument("video", type=str, help="path to the input video file")
    parser.add_argument("--output", "-o", type=str, required=False)
    parser.add_argument("--tone", "-t", type=str, required=True)
    parser.add_argument("--ffmpeg", type=str)
    parser.add_argument("--pitch_width", type=int, default=None)
    parser.add_argument("--pitch_position", type=str, default="top_right",
                        choices=["top_right", "top_left", "bottom_right", "bottom_left"])
    parser.add_argument("--min_pitch", type=str, default="D2")
    parser.add_argument("--max_pitch", type=str, default="G5")

    options = parser.parse_args()

    if not os.path.exists(options.audio):
        print(f"Audio file {options.audio} does not exist")
        exit(1)

    if not os.path.exists(options.video):
        print(f"Video file {options.video} does not exist")
        exit(1)

    if options.ffmpeg is None:
        options.ffmpeg = shutil.which("ffmpeg")
    if options.ffmpeg is None or not os.path.exists(options.ffmpeg):
        print("Unable to locate ffmpeg, use --ffmpeg to specify the path to ffmpeg")
        exit(1)

    if options.tone not in TONES:
        print(f"Invalid tone {options.tone}")
        exit(1)

    if options.output is None:
        options.output = ".".join(options.video.split(".")[:-1]) + "_with_pitch.mp4"

    try:
        min_freq = Tonality.normalize_to_freq(options.min_pitch)
        max_freq = Tonality.normalize_to_freq(options.max_pitch)
    except Exception:
        print(f"Invalid min/max pitch {options.min_pitch}/{options.max_pitch}")
        exit(1)

    plt.rcParams['animation.ffmpeg_path'] = options.ffmpeg

    with tempfile.TemporaryDirectory() as tmpdir:
        pitch_video_path = os.path.join(tmpdir, "pitch.mp4")
        generate_pitch_video(options.audio, pitch_video_path, options.tone, min_freq, max_freq)
        combine_video(options.ffmpeg, options.video, pitch_video_path, options.output, options.pitch_width, options.pitch_position)
