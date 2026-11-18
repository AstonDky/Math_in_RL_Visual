# Application demonstration

[rl-demo.mp4](rl-demo.mp4) records the actual PyQt6 widgets and live `TrainingEngine`: start, pause, edit and save display preferences, resume, stop, load the checkpoint, and continue. It begins with the ready interface, without editor or startup footage. The lower caption band describes the scripted action; the panels show the application's own output.

The recording uses Q-learning with uniform random behavior, seed 7, a 5 × 5 GridWorld, and a 60-step episode limit. It is a short interaction demonstration, not a convergence or timing benchmark. The application uses temporary settings, checkpoints, and logs so recording does not alter a user's work. Event times and progress are in [rl-demo.json](rl-demo.json).

To record another demonstration, install the normal project requirements and FFmpeg, then run from the repository root:

```bash
python tools/record_demo.py --ffmpeg /path/to/ffmpeg
```

The script writes the MP4 here and updates the light/dark screenshots under `docs/assets/`. It captures the real widgets in an offscreen Qt session, including the visible settings dialog. A system font with Chinese glyphs is needed; on Windows the script loads available Microsoft YaHei fonts.

Original project recordings remain available: [Chinese README recording](https://github.com/user-attachments/assets/fc999b72-8a74-4f34-b75d-a07ffba51816), [English README recording](https://github.com/user-attachments/assets/d823af37-37c2-47f4-bcd5-6993d6432635). They show the earlier interface.

## 中文说明

新版演示直接从可操作界面开始，录下真实训练、暂停、设置保存、恢复、停止与读取存档的过程。底部文字说明脚本触发的操作，主体画面来自程序实际渲染。录像使用独立的临时设置和训练文件，不改动已有学习记录。

录像为 32 秒、20 fps，无音轨；它展示交互过程，不作为算法收敛或运行速度的证据。运行上面的命令即可重新录制，参数和事件记录见同目录 JSON。
