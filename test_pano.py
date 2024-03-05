"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os,numpy,pathlib, os.path
from PIL import Image
import math
import glob
    
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

def crop(im,height,width,nb_pix,imgwidth,imgheight,nb_y,nb_x): 
    for i in range(nb_x+1):
        if i == 0 :
            a = i
            c = height
        elif i == nb_x :
            a = imgheight- height
            c = imgheight
        else:
            a = c-nb_pix
            c = a + height
        # print(i,a,c)
        for j in range(nb_y+1):
            if j == 0 :
                b = j
                d = width
            elif j == nb_y :
                b = imgwidth - width
                d = imgwidth
            else :
                b = d - nb_pix
                d = b + width
            # print(j,b,d)
            # if j == nb_y :
                # exit()
            box = (b, a, d, c)
            yield im.crop(box)
    
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concat_y(img_final,img,height,height_im):
    [a,b,c,d]=img.shape
    x1 = 0
    x2 = height
    x3 = height-20
    x4 = (2*height)-20 
    ad_image_y(img_final,img[:,:,:,0],img[:,:,:,1],x1,x2,x4,20)
    for i in range (2,d) :
        # print(i)
        if i == d-1 :
            # print('hi')
            x4 = height_im
            x2 =  x4 - 236
        else :
            x2 = x4
            x3 = x4-20
            x4 = x3+height
        ad_image_y(img_final,img_final,img[:,:,:,i],x1,x2,x4,20)        
    return img_final

def ad_image_y(img,img1,img2,x1,x2,x4,nb_pix):
    [height,width,col]=img1.shape
    img[x1:x2,:,:]=img1[x1:x2,:,:]
    img[x2:x4,:,:]=img2[nb_pix:height,:]
    for j in range(nb_pix+1) :
        w1 = (j/nb_pix)
        w2 = (1-(j/nb_pix))
        img[x2-1-j,:,:]=numpy.average( numpy.array([ img1[x2-1-j,:,:],img2[nb_pix-j,:,:]]), axis=0 , weights=[w1,w2])
    return img
def ad_image_x(img,img1,img2,x1,x2,y1,y2,y4,nb_pix,width):
    img[x1:x2,y1:y2,:]=img1[x1:x2,y1:y2,:]
    # print(img[x1:x2,y2:y4,:].shape,img2[:,nb_pix:width].shape, y2)
    img[x1:x2,y2:y4,:]=img2[:,nb_pix:width]        
    for j in range(nb_pix+1) :
        w1 = (j/nb_pix)
        w2 = (1-(j/nb_pix))
        img[x1:x2,y2-1-j,:]=numpy.average( numpy.array([ img1[:,y2-1-j,:],img2[:,nb_pix-j,:]]), axis=0 , weights=[w1,w2])
    return img

def concat_x(img, lst_im,nb_im,height,width,width_im):
    # print(nb_im,width_im)
    # exit()
    img0 = numpy.array(Image.open(lst_im[0]))
    img1 = numpy.array(Image.open(lst_im[1]))
    # print(lst_im[0],lst_im[1])
    x1 = 0
    x2 = height
    y1 = 0 
    y2 = width
    y3 = width-20
    y4 = (2*width)-20
    ad_image_x(img,img0,img1,x1,x2,y1,y2,y4,20,width)
    for i in range (2,nb_im) :
        if i == nb_im-1 :
            y4 = width_im
            y2 = y4 - 236 
        else :
            y2 = y4
            y3 = y4-20
            y4 = y3+width
        img1 = numpy.array(Image.open(lst_im[i]))
        # print(lst_im[i])
        ad_image_x(img,img,img1,x1,x2,y1,y2,y4,20,width)
    return img

if __name__ == '__main__':
    files = glob.glob('./Pano/*')

    files_test = glob.glob('./datasets/tapisserie/test/*')
    for f_test in files_test:
        os.remove(f_test)   
    files_image = glob.glob('./results/tapisserie/test_latest/images/*')
    for f_image in files_image:
        os.remove(f_image)   

    # print(files)
    for infile in files:
        # print(infile)
        height=256
        width=256
        start_num=0

        im = Image.open(infile).convert('L')
        imgwidth,imgheight = im.size

        nb_pix = 20

        nb_y_bis =imgwidth/(width-nb_pix)
        nb_y = math.ceil(imgwidth//(width-nb_pix))

        if nb_y_bis-nb_y<0.1 :
            nb_y =nb_y-1
        
        nb_x= math.ceil(imgheight//(height-nb_pix))
        # print(imgwidth, imgheight, 'nb_y', nb_y, 'nb_y_bis', nb_y_bis , 'nb_x',nb_x)
        for k,piece in enumerate(crop(im,height,width,nb_pix,imgwidth,imgheight,nb_y,nb_x),start_num):
            img=Image.new('RGB', (height,width), 255)
            img.paste(piece)
            path=os.path.join('./datasets/tapisserie/test',"IMG-{:03d}.png".format(k))

            get_concat_h(img, img).save(path)


        opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        opt.num_threads = 0   # test code only supports num_threads = 1
        opt.batch_size = 1    # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        webpage.save()  # save the HTML

        nb_y = nb_y +1
        nb_x = nb_x +1
        img_1 = numpy.zeros((height,imgwidth,3,nb_x))
        img_2 = numpy.zeros((imgheight,imgwidth,3))
        # print(nb_x,nb_y,imgwidth,imgheight)      
        newpath = './results/tapisserie/test_latest/images'
        # that directory
        images = sorted(pathlib.Path(newpath).glob('*[!_ifft]_fake_B.png'))
        for heiht_lvl in range(nb_x):
            # print(images[heiht_lvl*nb_y:(heiht_lvl+1)*nb_y])
            concat_x(img_1[:,:,:,heiht_lvl],images[heiht_lvl*nb_y:(heiht_lvl+1)*nb_y],nb_y,height,width,imgwidth)
        concat_y(img_2,img_1,height,imgheight)

        img = Image.fromarray(img_2.astype(numpy.uint8))
        path=infile.replace('E5/Pano','E5/results/Pano')
        img.save(path)
        
        files_test = glob.glob('./datasets/tapisserie/test/*')
        for f_test in files_test:
            os.remove(f_test)   
        files_image = glob.glob('./results/tapisserie/test_latest/images/*')
        for f_image in files_image:
            os.remove(f_image)   

