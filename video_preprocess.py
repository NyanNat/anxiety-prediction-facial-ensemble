import moviepy.editor as moviepy
import glob
import os

directory = []

dir = 'D:/Sekolah/111-1/Project/dataset'
directory = os.listdir(dir)

for folder in directory:
    files = os.listdir(dir+'/'+folder)
    for file in files:
        if file.endswith('.avi'):
            clip = moviepy.VideoFileClip(dir+'/'+folder+'/'+file)
            for i in range(0,6):
                cutdown = clip.subclip(i*30, i*30+30)
                cutdown.write_videofile(dir+'/'+folder+'/'+file[:-4]+'-'+str(i+1)+".mp4")
            os.remove(dir+'/'+folder+'/'+file)
        if file.endswith('.mp4'):
            if file[-8:-4] == 'T2-1' or file[-8:-4] == 'T3-1':
                os.remove(dir+'/'+folder+'/'+file)
