"""
Adapted from https://github.com/awsdang/360ImageConverterforColmap
"""

import os
import numpy as np
from PIL import Image
import py360convert
import concurrent.futures
import argparse
import time

class Converter:
    def __init__(self, input_dir:str, output_dir:str, resolution:tuple, base_angle:tuple, fov:tuple, overlap:int, threads:int, exclude_h_angles:list, exclude_v_angles:list, sort_v:bool=False, test_mode:bool=False, test_count:int=1):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.resolution = resolution
        self.base_angle=base_angle
        self.fov = fov
        self.h_fov, self.v_fov = fov
        self.threads = threads
        self.overlap = overlap
        self.exclude_h_angles = exclude_h_angles
        self.exclude_v_angles = exclude_v_angles
        self.sort_v = sort_v
        self.test_mode = test_mode
        self.test_count = test_count
        self.start_time = time.time()
        self.make_dir()
        self.images=self.assign_images()
        
    def print_time_to_finish(self)->None:
        end_time = time.time()
        print(f"{round(end_time-self.start_time, 2)} seconds.")

    def sort_angles(self, angles:list)->list:
        high_cutoff = max(self.exclude_h_angles) if self.exclude_h_angles else 360
        low_cutoff = min(self.exclude_h_angles) if self.exclude_h_angles else 0
        if(self.sort_v):
            x=0
            y=1
        else:
            x=1
            y=0
        return sorted(angles, key=lambda angle: (angle[x], angle[y] - 360 if angle[0] > high_cutoff else (angle[0] if angle[0] < low_cutoff else 360)))

    def genrate_steps(self)->tuple:
        h_step = self.h_fov * (1 - self.overlap / 100)
        v_step = self.v_fov * (1 - self.overlap / 100)
        return h_step, v_step

    def print_test_angles(self)->None:
        if(self.test_mode):
            for index, (h_angle, v_angle) in enumerate(self.angles):
                print(f"{index+1}: h:{h_angle} v:{v_angle}")

    def generate_viewing_angles(self)->list:
        h_step, v_step = self.genrate_steps()
        base_u, base_v = self.base_angle
        horizontal_angles = np.arange(base_u, 360, h_step) % 360
        vertical_start = max(-90, base_v - self.v_fov / 2)
        vertical_end = 90
        vertical_angles = np.arange(vertical_start, vertical_end + 1, v_step)
        arranged_vertical_angles = np.clip(vertical_angles, -90, 90)
        angles = [(h, v) for v in arranged_vertical_angles if v not in self.exclude_v_angles for h in horizontal_angles if h not in self.exclude_h_angles]  
        angles += [(h, 90) for h in horizontal_angles if h not in self.exclude_h_angles if 90 not in self.exclude_v_angles]
        self.angles = self.sort_angles(angles)

    def process_single_angle(self, filename:str, i:int, e_img)->None:
        u_deg, v_deg = self.angles[i]
        p_img = py360convert.e2p(e_img, fov_deg=self.fov, u_deg=u_deg, v_deg=v_deg, out_hw=self.resolution)
        formatted_i = f'{i+1:02d}'
        output_filename = f'{os.path.splitext(filename)[0]}_perspective_{formatted_i}.png'
        Image.fromarray(p_img).save(os.path.join(self.output_dir, output_filename))  

    def generate_images(self, filename:str, e_img):
        if self.test_mode:      
             with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                futures = [executor.submit(self.process_single_angle, filename, i, e_img) 
                           for i, (u_deg, v_deg) in enumerate(self.angles)]
                concurrent.futures.wait(futures)
        else:
            for i, (u_deg, v_deg) in enumerate(self.angles):
                self.process_single_angle(filename, i, e_img)
 
    def process_image(self, filename:str):
        file_path = os.path.join(self.input_dir, filename)
        print(f'Processing {filename}...', end=' ', flush=True)
        try:
            e_img = np.array(Image.open(file_path))
            self.generate_viewing_angles()
            self.print_test_angles()
            self.generate_images(filename, e_img)
            print(f'Processed {filename}')
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def make_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def assign_images(self):
        return [filename for filename in os.listdir(self.input_dir) if (filename.endswith('.jpg') or filename.endswith('.png'))]

    def init_process(self):
        if self.test_mode:
            images = self.images[:self.test_count]
            for filename in images: 
                self.process_image(filename)
        else:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = [executor.submit(self.process_image, filename)
                            for filename in self.images]
                    concurrent.futures.wait(futures)
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert 360 images to perspective views.')
    parser.add_argument('-i', '--input_directory', required=True, help='Input directory containing 360 images.')
    parser.add_argument('-o', '--output_directory', required=True, help='Output directory for perspective images.')
    parser.add_argument('-res','--resolution' , nargs=2, type=int, default=(1600, 1600), help='Output height and width of the images.')
    parser.add_argument('--base_angle', nargs=2, type=int, default=(0, 0), help='Base viewing angle for output images (u, v).')
    parser.add_argument('--fov', nargs=2, type=float, default=(90, 90), help='Field of view in degrees (horizontal, vertical).')
    parser.add_argument('--threads', type=int, default=4, help='Maximum number of worker threads.')
    parser.add_argument('--overlap', type=float, default=10, help='the overlaps parameter between images min 0, max 100.')
    parser.add_argument('--exclude_h_angles', type=str, default='', help='Comma-separated list of horizontal angles to exclude (e.g., "40,60,90"')
    parser.add_argument('--exclude_v_angles', type=str, default='', help='Comma-separated list of vertical angles to exclude (e.g., "40,60,90"')
    parser.add_argument('--sort_v', action='store_true', default=False, help='If specified, sort images vertically.')
    parser.add_argument('--test', action='store_true', default=False, help='If specified, process only the first image.')
    parser.add_argument('--test_count', type=int, default=None, help='If specified, process the number of frames')
    args = parser.parse_args()
    exclude_h_angles = [float(angle.strip()) for angle in args.exclude_h_angles.split(',') if angle.strip()] if args.exclude_h_angles else []
    exclude_v_angles = [float(angle.strip()) for angle in args.exclude_v_angles.split(',') if angle.strip()] if args.exclude_v_angles else []
    print(f"360-to-persp->input_dir{args.input_directory}, output_dir{args.output_directory}")
    converter = Converter(args.input_directory,
                          args.output_directory,
                          tuple(args.resolution),
                          tuple(args.base_angle),
                          tuple(args.fov),
                          args.overlap,
                          args.threads,
                          exclude_h_angles,
                          exclude_v_angles,
                          args.sort_v,
                          args.test,
                          args.test_count)
    converter.init_process()
    converter.print_time_to_finish()
    print("Success!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
