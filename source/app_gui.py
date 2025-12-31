# app_gui.py
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from pose_video_ios_app import process_video_app
from resources import ensure_dir

class PoseLabGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PoseLab 视频姿态分析")
        self.geometry("680x420")

        self.video_path = tk.StringVar()
        self.out_dir = tk.StringVar(value=os.path.abspath("output"))
        self.device = tk.StringVar(value="cpu")

        self._build()

    def _build(self):
        pad = {"padx": 10, "pady": 8}

        frm = tk.Frame(self)
        frm.pack(fill="both", expand=True)

        tk.Label(frm, text="输入视频：").grid(row=0, column=0, sticky="w", **pad)
        tk.Entry(frm, textvariable=self.video_path, width=60).grid(row=0, column=1, sticky="we", **pad)
        tk.Button(frm, text="选择…", command=self.pick_video).grid(row=0, column=2, **pad)

        tk.Label(frm, text="输出目录：").grid(row=1, column=0, sticky="w", **pad)
        tk.Entry(frm, textvariable=self.out_dir, width=60).grid(row=1, column=1, sticky="we", **pad)
        tk.Button(frm, text="选择…", command=self.pick_outdir).grid(row=1, column=2, **pad)

        tk.Label(frm, text="设备：").grid(row=2, column=0, sticky="w", **pad)
        tk.Entry(frm, textvariable=self.device, width=20).grid(row=2, column=1, sticky="w", **pad)
        tk.Label(frm, text="(cpu / cuda:0 / mps)").grid(row=2, column=1, sticky="e", **pad)

        self.run_btn = tk.Button(frm, text="开始分析", command=self.run)
        self.run_btn.grid(row=3, column=1, sticky="w", **pad)

        self.log_box = tk.Text(frm, height=14, width=80)
        self.log_box.grid(row=4, column=0, columnspan=3, sticky="nsew", **pad)

        frm.grid_columnconfigure(1, weight=1)
        frm.grid_rowconfigure(4, weight=1)

    def pick_video(self):
        p = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")]
        )
        if p:
            self.video_path.set(p)

    def pick_outdir(self):
        p = filedialog.askdirectory(title="选择输出目录")
        if p:
            self.out_dir.set(p)

    def log(self, msg: str):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    def run(self):
        v = self.video_path.get().strip()
        outdir = self.out_dir.get().strip()
        dev = self.device.get().strip()

        if not v or not os.path.exists(v):
            messagebox.showerror("错误", "请先选择一个有效的视频文件。")
            return

        ensure_dir(outdir)
        self.run_btn.config(state="disabled")
        self.log_box.delete("1.0", "end")
        self.log("[INFO] 开始处理…")

        def worker():
            try:
                process_video_app(
                    video_path=v,
                    out_dir=outdir,
                    device=dev,
                    kpt_thr=0.3,
                    log_cb=self.log,
                )
                messagebox.showinfo("完成", f"处理完成！\n输出目录：{outdir}")
            except Exception as e:
                messagebox.showerror("失败", str(e))
            finally:
                self.run_btn.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    PoseLabGUI().mainloop()