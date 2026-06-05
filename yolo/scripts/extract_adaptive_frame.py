# 需要先安装：pip install opencv-python numpy，之后直接运行该文件即可：python extract_adaptive_frame.py，或点击运行键
import cv2
import os
import numpy as np

def calc_frame_diff(prev_frame, cur_frame):
    # 计算两帧画面像素差异，数值越大画面变化越大
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return np.var(diff)

def extract_adaptive_frame(vid_path: str, save_dir: str, min_pick=5, max_pick=10, var_thresh=1300.0):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"打开视频失败:{vid_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = int(total_frames / fps)
    img_num = 1
    prefix = save_dir.split('_')[0]

    for sec in range(total_sec):
        sec_start = int(sec * fps)
        sec_end = int((sec+1)*fps)
        sec_end = min(sec_end, total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, sec_start)
        _, frame_s = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec_end-1)
        _, frame_e = cap.read()

        diff_score = calc_frame_diff(frame_s, frame_e)
        norm_score = np.clip(diff_score / var_thresh, 0, 1)
        pick_count = int(min_pick + norm_score * (max_pick - min_pick))
        print(f"第{sec}秒 | 方差:{diff_score:.1f} | 阈值{var_thresh} | 每秒抽取:{pick_count}张")

        sample_pos = np.linspace(sec_start, sec_end-1, pick_count, dtype=int)
        for pos in sample_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, img = cap.read()
            if not ret:
                continue
            save_name = os.path.join(save_dir, f"{prefix}{img_num:02d}.jpg")
            cv2.imwrite(save_name, img)
            img_num += 1

    cap.release()
    print(f"\n[{vid_path}] 完成，总共输出 {img_num-1} 张图片 → {save_dir}\n")

if __name__ == "__main__":
    extract_adaptive_frame("train.mp4", "train_imgs", var_thresh=1300.0)
    extract_adaptive_frame("test.mp4", "test_imgs", var_thresh=1300.0)