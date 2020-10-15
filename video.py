import os

from numpy.lib.type_check import real_if_close
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from fast_style_transfer.net import Net
from fast_style_transfer.option import Options
import fast_style_transfer.utils
from fast_style_transfer.utils import StyleLoader
from face_parsing.model import BiSeNet


def resize(size, h, w):
    if h > w:
        h = size * h/w
        w = size 
    else:
        w = size * w/h 
        h = size
    return int(h), int(w)


def run_demo(args, mirror=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Face Parsing
    bisenet = BiSeNet(n_classes=19)
    bisenet.load_state_dict(torch.load('face_parsing/res/79999_iter.pth', map_location=device))
    bisenet.to(device).eval()

    style_model = Net(ngf=args.ngf)
    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
    	if key.endswith(('running_mean', 'running_var')):
    		del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model.eval()
    if args.cuda:
        style_loader = StyleLoader(args.style_folder, args.style_size)
        style_model.cuda()
    else:
        style_loader = StyleLoader(args.style_folder, args.style_size, False)

	# Define the codec and create VideoWriter object
    if args.video_file is None:
        cam = cv2.VideoCapture(0)
        is_cam = True
    else:
        cam = cv2.VideoCapture(args.video_file)
        is_cam = False

    w, h = cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height, width = resize(args.demo_size, h, w)

    swidth = int(width/4)
    sheight = int(height/4)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    key = 0
    idx = 0
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if is_cam:
            width, height = cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)        
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(width), int(height)))

    chosens = [[0, 16], [17]]
    styles = [args.style_a]
    if args.style_b is not None:
        styles.append(args.style_b)

    while cam.isOpened():
        # read frame
        idx += 1
        ret_val, img = cam.read()
        if not ret_val:
            break
        if not is_cam:
            img = cv2.resize(img, (width, height))
        if mirror:
            img = cv2.flip(img, 1)
        cimg = img.copy()
        img = np.array(img).transpose(2, 0, 1)

        the_image = img.copy()

        stylizeds = []
        for i in range(len(styles)):

            style_v = style_loader.get(styles[i])
            style_v = Variable(style_v.data)
            style_model.setTarget(style_v)

            img=torch.from_numpy(the_image.copy()).unsqueeze(0).float()
            if args.cuda:
                img=img.cuda()

            img = Variable(img)
            img = style_model(img)

            if args.cuda:
                #simg = style_v.cpu().data[0].numpy()
                img = img.cpu().clamp(0, 255).data[0].numpy()
            else:
                #simg = style_v.data.numpy()
                img = img.clamp(0, 255).data[0].numpy()
            #simg = np.squeeze(simg)

            img = img.transpose(1, 2, 0).astype('uint8')
            stylizeds.append(img.copy())

        img = Image.fromarray(cv2.cvtColor((cimg).astype(np.uint8), cv2.COLOR_BGR2RGB))

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            img = to_tensor(img)
            img = torch.unsqueeze(img, 0)
            if torch.cuda.is_available():
                img = img.cuda()
            out_net = bisenet(img)[0]
            segments = out_net.squeeze(0).cpu().numpy()
        temperature = args.tau
        ex = segments*temperature
        ex = ex - np.max(ex, axis=0)
        denominator_softmax = np.sum(np.exp(ex), axis=0)

        img = cimg.copy()/255
        for i in range(len(chosens)):
            alpha = np.zeros(segments[0].shape)
            for s in chosens[i]:
                alpha += np.exp(ex[s]) / denominator_softmax
            alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            img = (stylizeds[i]/255)*alpha + (1-alpha)*(img)
            
        img = (img*255).astype('uint8')

        #img = img.transpose(1, 2, 0).astype('uint8')
        #simg = simg.transpose(1, 2, 0).astype('uint8')

		# display
        #simg = cv2.resize(simg,(swidth, sheight))
        #cimg[0:sheight,0:swidth,:]=simg
        #img = np.concatenate((cimg,img),axis=1)
        cv2.imshow('MSG Demo', img)
		#cv2.imwrite('stylized/%i.jpg'%idx,img)
        key = cv2.waitKey(1)
        if args.record:
        	out.write(img)
        if key == 27: 
        	break
    cam.release()
    if args.record:
        out.release()
    cv2.destroyAllWindows()

def main():
	# getting things ready
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")

	# run demo
	run_demo(args, mirror=True)

if __name__ == '__main__':
	main()
