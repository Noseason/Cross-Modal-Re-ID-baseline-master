import numpy as np
from PIL import Image
import torch.utils.data as data
import os


class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        data_dir = '../Datasets/SYSU-MM01/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        data_dir = '/data3/QK/remote/AGW/datasets/RegDB/'
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

#######以下改成tvpr2的处理
class TVPRData(data.Dataset):
    ###改成传入两个transform
    def __init__(self, data_dir, trial, transform1=None,transform2=None, colorIndex=None, depthIndex=None):
        # Load training images (path) and labels
        #train要大概五百个人，每个人color和depth数量一致，但人和人之间数量不一样

        train_color_list = []
        train_depth_list = []
        #先读取color里面的train，depth里面的train里的所有图片，编成list
        data_dir_color = '/data3/QK/TransREID/TransReID-main/data/color/market1501/bounding_box_train'
        data_dir_depth = '/data3/QK/TransREID/TransReID-main/data/depth/market1501/bounding_box_train'

        for root, dirs, files in os.walk(data_dir_color):
            if root != data_dir_color:
                break
            for file in files:
                path = os.path.join(root, file)
                train_color_list.append(path)

        for root, dirs, files in os.walk(data_dir_depth):
            if root != data_dir_depth:
                break
            for file in files:
                path = os.path.join(root, file)
                train_depth_list.append(path)

        color_img_file, train_color_label = load_data1(train_color_list)
        depth_img_file, train_depth_label = load_data1(train_depth_list)

        print("len(color_img_file)")
        print(len(color_img_file))
        print("len(depth_img_file)")
        print(len(depth_img_file))

        train_color_image = []
        for i in range(len(color_img_file)):
            print("color",i)
            img = Image.open(color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_depth_image = []
        for i in range(len(depth_img_file)):
            print("depth", i)
            img = Image.open(depth_img_file[i]).convert('L')
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_depth_image.append(pix_array)
        train_depth_image = np.array(train_depth_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_depth_image = train_depth_image
        self.train_depth_label = train_depth_label

        self.transform1 = transform1
        self.transform2 = transform2

        self.cIndex = colorIndex
        self.dIndex = depthIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_depth_image[self.dIndex[index]], self.train_depth_label[self.dIndex[index]]

        img1 = self.transform1(img1)
        img2 = self.transform2(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class TVPRData11(data.Dataset):##############新的，浪潮用
    ###改成传入两个transform
    def __init__(self, data_dir, trial, transform1=None,transform2=None, colorIndex=None, depthIndex=None):
        # Load training images (path) and labels
        #train要大概五百个人，每个人color和depth数量一致，但人和人之间数量不一样
        ###从pair1000直接读取，pair1000/12/123_color.png,123_depth.png
        train_color_list = []###train的彩色图片的路径集合
        train_depth_list = []

        data_dir = '/data3/QK/REID/small1000'

        # for root,dirs, files in os.walk(data_dir):
        #     dirs.sort(key=lambda x: int(x))
        #     print(dirs)
        #     print(len(dirs) // 2 + 1)
        #     for i in range(0, len(dirs) // 2 + 1):
        #         print(i)
        #         print("dirs[i]")
        #         print(dirs[i])
        #         path0 = os.path.join(root,dirs[i])####pair1000/12
        #         for _,_, files1 in os.walk(path0):
        #             for file in files1:
        #                 if file.endswith("color.png"):
        #                     path = os.path.join(path0,file)
        #                     train_color_list.append(path)
        #                 else:
        #                     path = os.path.join(path0, file)
        #                     train_depth_list.append(path)
        dirs = []
        for dir in os.listdir(data_dir):
            dirs.append(dir)
        dirs.sort(key=lambda x: int(x))
        #print(dirs)
        #print(len(dirs) // 2 + 1)
        for i in range(0, len(dirs) // 2 + 1):
            # print(i)
            # print("dirs[i]")
            # print(dirs[i])
            path0 = os.path.join(data_dir,dirs[i])####pair1000/12
            for file in os.listdir(path0):

                if file.endswith("color.png"):
                    path = os.path.join(path0,file)
                    train_color_list.append(path)
                elif file.endswith("depth.png"):
                    path = os.path.join(path0, file)
                    train_depth_list.append(path)

        #print("len(train_color_list)")
        #print(len(train_color_list))

        print("first")
        color_img_file, train_color_label = load_data2(train_color_list)
        depth_img_file, train_depth_label = load_data2(train_depth_list)
        print("second")

        print("train color label 1-10")
        for i in range(0,10):
            print(train_color_label[i])
        print(train_color_label[100])

        #print(color_img_file)
        print("len(color_img_file):")
        print(len(color_img_file))
        #print(depth_img_file)
        print("len(depth_img_file):")
        print(len(depth_img_file))

        train_color_image = []
        for i in range(len(color_img_file)):
            #print(i)
            img = Image.open(color_img_file[i])
            #print("third")
            img = img.resize((144, 288), Image.ANTIALIAS)
            #print("fourth")
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)
        #print("third")
        train_depth_image = []
        for i in range(len(depth_img_file)):
            img = Image.open(depth_img_file[i]).convert('L')
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_depth_image.append(pix_array)
        train_depth_image = np.array(train_depth_image)
        print("fourth")
        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_depth_image = train_depth_image
        self.train_depth_label = train_depth_label

        self.transform1 = transform1
        self.transform2 = transform2

        self.cIndex = colorIndex
        self.dIndex = depthIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_depth_image[self.dIndex[index]], self.train_depth_label[self.dIndex[index]]

        img1 = self.transform1(img1)
        img2 = self.transform2(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):
#####这边transform要分情况改
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

def load_data1(data_file_list):
    file_label = [int(s.split('/')[-1].split('_')[0]) for s in data_file_list]
    return data_file_list, file_label


####浪潮
def load_data2(data_file_list):
    ###pair1000/12/xxx_color.png
    file_label = [int(s.split('/')[-2]) for s in data_file_list]
    return data_file_list, file_label