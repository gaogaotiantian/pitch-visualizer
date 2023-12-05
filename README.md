# Pitch Visualizer

这是一个小小的Python脚本，可以把一段纯人声音频进行音高的可视化，然后和音频的原视频进行合并，
最终生成一段带有动态音高标注的视频。

## 开始之前

### FFmpeg

你需要[ffmpeg]https://ffmpeg.org/。请确保在命令行输入`ffmpeg`可以正常调用ffmpeg，或者
你知道明确的ffmpeg的位置。

### Python requirements

建议大家在[虚拟环境](https://docs.python.org/zh-cn/3/library/venv.html)中工作

```
python -m venv venv
source venv/bin/activate
```

最简单的方式：

```
pip install -r requirements.txt
```

### 人声分离

这个脚本并不能帮你做人声分离，你需要用其他的软件或者库来完成这件事。

* [Ultimate Vocal Remover](https://ultimatevocalremover.com/)是目前我用过效果最好的软件，免费
* [spleeter](https://github.com/deezer/spleeter)是一个开源的人声分离库

## 使用

你需要准备两个文件

* 原始视频文件（.mp4）
* 从原始视频中分离的干声文件（.mp3）

为了画出准确的对比音高，你需要知道这首歌曲的调性（C或者F#之类）。在这个软件中，我们没有特意区别小调，
调性只是为了标注音阶。如果歌曲是小调，请使用其音阶对应的大调的调性。比如A小调对应的是C。

```
python gen_pitch.py --audio <voice.mp3> -t <tone> <video.mp3> -o <output.mp4>

# 举个例子
python gen_pitch.py --audio wjk_raw.mp3 -t E wjk.mp4 -o wjk_with_pitch.mp4
```

其中`-o`可以省略，默认是在输入视频的文件夹下建立另一个视频。

### 其他的选项

* `--ffmpeg`可以指定你想用的ffmpeg，在你PATH中不存在ffmpeg的时候需要使用
* `--pitch_width`设置音高图的宽度，默认是原始视频的一半
* `--pitch_position`设置音高图的位置，默认是`top_right`

## LICENSE

Copyright 2023 Tian Gao.

Distrubuted under the terms of the [Apache 2.0 license](https://github.com/gaogaotiantian/pitch-visualizer/blob/master/LICENSE)
