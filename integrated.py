import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def preprocess_command():
    all_images = os.listdir('./images')
    all_combinations = combinations(all_images, 2)
    st = ''
    for i in all_combinations:
        st += str(i[0]) + " " + str(i[1]) + ' 0 0 1163.45 0. 653.626 0. 1164.79 481.6 0. 0. 1. 1163.45 0. 653.626 0. 1164.79 481.6 0. 0. 1. 0.78593 -0.35128 0.50884 -1.51061 0.39215 0.91944 0.02904 -0.05367 -0.47805 0.17672 0.86037 0.056 0. 0. 0. 1.\n'
    with open('command.txt', 'w') as f:
        f.write(st)



def execute_model(viz = True, resize = True):
    #by default the model resized the images to 640x480
    if(viz == True and resize == True):
        os.system('python match_pairs.py --input_pairs ./command.txt --input_dir ./images --superglue outdoor --viz')
    elif(viz == False and resize == True):
        os.system('python match_pairs.py --input_pairs ./command.txt --input_dir ./images --superglue outdoor')        
    elif(viz == True and resize == False):
        os.system('python match_pairs.py --input_pairs ./command.txt --input_dir ./images --superglue outdoor --viz --resize -1')
    else:
        os.system('python match_pairs.py --input_pairs ./command.txt --input_dir ./images --superglue outdoor --resize -1')





if __name__ == "__main__":
    preprocess_command()
    execute_model(viz = False, resize=False)