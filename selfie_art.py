import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from face_parsing.model import BiSeNet
from style_transfer.style_transfer import run_style_transfer
from fast_style_transfer.utils import tensor_load_rgbimage2, preprocess_batch
from fast_style_transfer.net import Net, Variable
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def rescale(i, size):
    h, w = i.size
    if h > w:
        h = size * h/w
        w = size 
    else:
        w = size * w/h 
        h = size
    return i.resize((int(h),int(w)), Image.ANTIALIAS)

class SelfieArtCore:

    def __init__(self):
        self.image = None
        self.style_image = None
        self.stylized_image = None
        self.segmentation_mask = None
        self.result_image = None

        self.img_size = 512

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Face Parsing
        self.bisenet = BiSeNet(n_classes=19)
        self.bisenet.load_state_dict(torch.load('face_parsing/res/79999_iter.pth', map_location=self.device))
        self.bisenet.to(self.device).eval()

        # Style Transfer
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
    def set_image(self, image_path, run_face_parsing=True):
        self.image = Image.open(image_path)
        self.image = rescale(self.image, self.img_size)
        self.stylized_image = None
        self.result_image = self.image.copy()
        if run_face_parsing:
            self._run_face_parsing()
    
    def reset(self):
        self.result_image = self.image.copy()
    
    def set_style(self, image_path):
        self.stylized_image = None
        self.style_image = Image.open(image_path).resize(self.image.size, Image.BILINEAR)

    def save_result(self, filename):
        if self.result_image is not None:
            self.result_image.save(filename)

    def get_soft_mask(self, segments, temperature):
        alpha = np.zeros(self.segments[0].shape)
        if temperature != 100.0:
            ex = self.segments*temperature
            ex = ex - np.max(ex, axis=0)
            denominator_softmax = np.sum(np.exp(ex), axis=0)
            for s in segments:
                alpha += np.exp(ex[s]) / denominator_softmax
        else:
            for s in segments:
                alpha[self.segmentation_mask == s] = 1
        return alpha

    def apply_color(self, rgb, segments, temperature=1, over=False):
        r, g, b = rgb
        if over:
            image = cv2.cvtColor(np.array(self.image.copy()), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.cvtColor(np.array(self.result_image.copy()), cv2.COLOR_RGB2BGR)
        tar_color = np.zeros_like(image)
        tar_color[:, :, 0] = b
        tar_color[:, :, 1] = g
        tar_color[:, :, 2] = r
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
        changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR) /255.
        image = image /255.

        alpha = self.get_soft_mask(segments, temperature)

        #plt.figure()
        #plt.imshow(alpha.copy(), vmin=0, vmax=1)
        #plt.axis('off')
        #plt.savefig(f'out{temperature}.png', bbox_inches='tight', pad_inches=0)
        #plt.show()

        #plt.figure()
        #sns.distplot(alpha.copy().flatten(), color='blue',  kde_kws={"bw": .15}, hist_kws={"alpha": 1, "color": "b"})
        #sns.distplot(alpha.copy().flatten(), color='blue',  kde=False, norm_hist=True, hist_kws={"alpha": 1, "color": "b"})
        #plt.xlim([0,1])
        #plt.xticks(fontsize=14)
        #plt.yticks([])
        #plt.savefig(f'dist{temperature}.png', bbox_inches='tight', pad_inches=0)
        #plt.show()

        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        image = alpha*changed + (1-alpha)*image
        self.result_image = Image.fromarray(cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    def apply_style(self, segments, num_iterations=300, temperature=1, fast=False, over=False):
        if self.style_image is None:
            return
        if self.stylized_image is None:
            if fast:
                self._run_fast_style_transfer()
            else:
                self._run_style_transfer(num_iterations)

        if over:
            res = cv2.cvtColor(np.array(self.image.copy()), cv2.COLOR_RGB2BGR) /255.
        else:
            res = cv2.cvtColor(np.array(self.result_image.copy()), cv2.COLOR_RGB2BGR) /255.
        style = cv2.cvtColor(np.array(self.stylized_image.copy()), cv2.COLOR_RGB2BGR) /255.
        
        alpha = self.get_soft_mask(segments, temperature)
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        res = alpha*style + (1-alpha)*res
        self.result_image = Image.fromarray(cv2.cvtColor((res*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    def _run_face_parsing(self):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            img = self.image.copy()
            #image = img.resize((512, 512), Image.BILINEAR)
            image = img
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            if str(self.device) == 'cuda':
                img = img.cuda()
            out = self.bisenet(img)[0]
            self.segments = out.squeeze(0).cpu().numpy()
            self.segmentation_mask = self.segments.argmax(0)

    def _run_style_transfer(self, num_iterations):
        loader = transforms.Compose([
            transforms.Resize(self.img_size),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor
        unloader = transforms.ToPILImage()  # reconvert into PIL image
        def image_loader(image):
            # fake batch dimension required to fit network's input dimensions
            image = loader(image).unsqueeze(0)
            return image.to(self.device, torch.float)

        content_image = image_loader(self.image.copy())
        input_image = content_image.clone()
        style_image = image_loader(self.style_image.copy())
        output = run_style_transfer(self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std, content_image, style_image, input_image, num_steps=num_iterations)
        image = output.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        self.stylized_image = unloader(image).resize(self.image.size, Image.BILINEAR)

    def _run_fast_style_transfer(self):
        content_image = tensor_load_rgbimage2(self.image, size=self.img_size, keep_asp=True)
        content_image = content_image.unsqueeze(0)
        style = tensor_load_rgbimage2(self.style_image, size=self.img_size)
        style = style.unsqueeze(0)    
        style = preprocess_batch(style)

        style_model = Net(ngf=128)
        model_dict = torch.load('./fast_style_transfer/models/21styles.model')
        model_dict_clone = model_dict.copy()
        for key, value in model_dict_clone.items():
            if key.endswith(('running_mean', 'running_var')):
                del model_dict[key]
        style_model.load_state_dict(model_dict, False)

        if str(self.device) == 'cuda':
            style_model.cuda()
            content_image = content_image.cuda()
            style = style.cuda()

        style_v = Variable(style)

        content_image = Variable(preprocess_batch(content_image))
        style_model.setTarget(style_v)

        output = style_model(content_image)
        (b, g, r) = torch.chunk(output.data[0], 3)
        output = torch.cat((r, g, b))
        if str(self.device) == 'cuda':
            img = output.clone().cpu().clamp(0, 255).numpy()
        else:
            img = output.clone().clamp(0, 255).detach().numpy()
        img = img.transpose(1, 2, 0).astype('uint8')
        self.stylized_image = Image.fromarray(img).resize(self.image.size, Image.BILINEAR)

if __name__ == '__main__':
    
    """sa = SelfieArtCore()
    sa.set_image('images/menina.jpg')
    sa.set_style('images/starry_night.jpg')
    sa._run_face_parsing()
    sa._run_style_transfer() """
