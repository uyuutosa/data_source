from PIL import Image
import numpy as np
import PIL.Image as I
import cv2
import random

class CorrectExif(object):
    """
    """

    def __init__(self, ):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """

        convert_image = {
            1: lambda img: img,
            2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            3: lambda img: img.transpose(Image.ROTATE_180),
            4: lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
            5: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Pillow.ROTATE_90),
            6: lambda img: img.transpose(Image.ROTATE_270),
            7: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Pillow.ROTATE_270),
            8: lambda img: img.transpose(Image.ROTATE_90),
            }

        if not '_getexif' in dir(img):
            return img
            
        exif = img._getexif()
        if exif is None:
            return img
        else:
            orientation = exif.get(0x112, 1)

            return convert_image[orientation](img)

class HumanCrop(object):
    """
    """

    def __init__(self, margin, weight_path="pose_model.pth", scale=0.5, gpu_ids=''):
        import aimaker.predictor.pose_predictor as pp
        self.margin = int(margin)
        self.scale = scale
        self.predictor = pp.PosePredictor(weight_path=weight_path, gpu_ids=gpu_ids)


    def _getCropInformation(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        img = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale)

        ann, ret = self.predictor.predict(img)
        clipped_img_lst = []
        x_lst = []
        y_lst = []
        for arr in ann:
            n, x, y = arr
            x_lst += [x]
            y_lst += [y]
        x_arr, y_arr = np.array(x_lst), np.array(y_lst)
        x_min, x_max, y_min, y_max = x_arr.min(), x_arr.max(), y_arr.min(), y_arr.max()
        x_min /= self.scale
        x_max /= self.scale
        y_min /= self.scale
        y_max /= self.scale
        return x_min, x_max, y_min, y_max

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        raw_img = img.copy()

        x_min, x_max, y_min, y_max = self._getCropInformation(img)
        width, height = x_max - x_min, y_max - y_min

        y_origin = y_min - self.margin // 2
        x_origin = x_min - self.margin // 2

        #x_origin /= self.scale
        #y_origin /= self.scale
        #width    /= self.scale
        #height   /= self.scale
        margin   =  int(self.margin / self.scale)
        x_origin, y_origin, width, height = int(x_origin), int(y_origin), int(width), int(height)


        return I.fromarray(raw_img[y_origin: y_origin + height + margin, x_origin: x_origin + width + margin])

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """

        if not isinstance(img, np.ndarray):
            img = np.array(img)

        raw_img = img.copy()
        img = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale)

        ann, ret = self.predictor.predict(img)
        clipped_img_lst = []
        x_lst = []
        y_lst = []
        for arr in ann:
            n, x, y = arr
            x_lst += [x]
            y_lst += [y]
        x_arr, y_arr = np.array(x_lst), np.array(y_lst)
        x_min, x_max, y_min, y_max = x_arr.min(), x_arr.max(), y_arr.min(), y_arr.max()
        width, height = x_max - x_min, y_max - y_min

        y_origin = y_min - self.margin // 2
        x_origin = x_min - self.margin // 2

        x_origin /= self.scale
        y_origin /= self.scale
        width    /= self.scale
        height   /= self.scale
        margin   =  int(self.margin / self.scale)
        x_origin, y_origin, width, height = int(x_origin), int(y_origin), int(width), int(height)


        return I.fromarray(raw_img[y_origin: y_origin + height + margin, x_origin: x_origin + width + margin])
        #return torch.Tensor(np.array(clipped_img_lst))

class DivideBodyRegion(object):
    """
    """

    def __init__(self, width, height, clip_number_lst, weight_path="pose_model.pth", gpu_ids=''):
        import aimaker.predictor.pose_predictor as pp
        self.width = width
        self.height = height
        self.clip_number_lst = clip_number_lst
        self.predictor = pp.PosePredictor(weight_path=weight_path, gpu_ids=gpu_ids)

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        ann, ret = self.predictor.predict(img)
        clipped_img_lst = []
        for arr in ann:
            n, x, y = arr
            if n in self.clip_number_lst:
                x_origin = x - self.width  // 2
                y_origin = y - self.height // 2
                clipped_img_lst += [img[y_origin:y_origin+self.height, x_origin:x_origin+self.width]]

        return torch.Tensor(clipped_img_lst)
        #return torch.Tensor(np.array(clipped_img_lst))
        

class BodygramDataTransform(object):
    """
    """
    def __init__(self, width, height, weight_path="pose_model.pth", scale=0.5, gpu_ids=''):
        import aimaker.predictor.pose_predictor as pp
        self.width = width
        self.height = height
        self.scale = scale
        self.predictor = pp.PosePredictor(weight_path=weight_path, gpu_ids=gpu_ids)


    def _generateGaussianMap(self, img):
        height, width, channel = img.shape

        def __pointandGenerateGaussian(x_0, y_0, width, height, sigma_x=100, sigma_y=100):
            y, x = np.mgrid[0:height, 0:width]
            return (255 * np.exp(-((x-x_0)**2 / sigma_x + (y - y_0)**2 / sigma_y))).astype(np.uint8)

        map_lst = []
        for arr in self.p_lst:
            x, y = arr
            map_lst += [__pointandGenerateGaussian(x, y, width, height)]

        return map_lst

    def __pointandGenerateGaussian(self, x_0, y_0, width, height, sigma_x=100, sigma_y=100):
        y, x = np.mgrid[0:height, 0:width]
        return (255 * np.exp(-((x-x_0)**2 / sigma_x + (y - y_0)**2 / sigma_y))).astype(np.uint8)

    def _rescale(self, x_origin, y_origin, width, height):
        x_origin /= self.scale
        y_origin /= self.scale
#        width    /= self.scale
#        height   /= self.scale
        return int(x_origin), int(y_origin), int(width), int(height)
        #x_origin, y_origin, width, height = int(x_origin), int(y_origin), int(width), int(height)

    def _correctShape(self, img):
        a,b,c = img.shape
        if a == 3:
            img = img.transpose(1,2,0)
        return img


    def __call__(self, img, p_lst, ann_lst):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """

        if not isinstance(img, np.ndarray):
            img = np.array(img)

        img = self._correctShape(img)

        raw_img = img.copy()
        img = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale)
        cv2.imwrite("tmp.jpg", img)
        ann, ret = self.predictor.predict(img)

        x_lst = []
        y_lst = []
        cropped_img_lst = []
        cropped_map_lst = []
        already_processed_lst = []
        cnt = 0
        for arr in ann:
            #print(arr)
            n, x, y = arr
            if n in ann_lst:
                if n in already_processed_lst:
                    continue

                already_processed_lst += [n]
                
                x_lst += [x]
                y_lst += [y]
                x_origin = x - self.width  // 4
                y_origin = y - self.height // 4
                # for raw image
                x_origin, y_origin, width, height = self._rescale(x_origin, y_origin, self.width, self.height)
                #print(x_origin, y_origin, width, height)
                cropped_img_lst += [raw_img[y_origin:y_origin+height, x_origin:x_origin+width]]
                c_lst = []
                for p in p_lst:
                    lst = [np.zeros((height, width))]
                    #print(x_origin, x_origin + width)
                    #print(y_origin, y_origin + height)
                    if x_origin <= p[0] and p[0] < x_origin + width and y_origin <= p[1] and p[1] < y_origin + height:
                        x_map, y_map = p[0] - x_origin, p[1] - y_origin
                        #print("map")
                        #print(x_map, y_map,)
                        img = self.__pointandGenerateGaussian(x_map, y_map, width, height)
                        I.fromarray(img).save("tmp%d.jpg" %cnt)
                        cnt += 1
                        lst += [img]
                        #print(lst[-1].max())
                    tmp = np.array(lst)
                    c_lst += [tmp.max(0)]
                cropped_map_lst += [c_lst]
                #print("map desu")
                #print(np.array(cropped_map_lst).shape)
            #input(cropped_map_lst)
            #cropped_map_lst += [np.array(c_lst).max(axis=0)]
        #print(cropped_map_lst)
        #for m in cropped_map_lst:
        #    print(np.array(m))
        #    print(m[0].shape)
        #print(torch.Tensor(np.array(cropped_map_lst)).shape)
        #print(torch.Tensor(np.array(cropped_img_lst)).shape)
        #print(torch.Tensor(cropped_img_lst).shape)
    
        #print(cropped_img_lst)
        #input("here")
        #print(torch.Tensor(np.array(cropped_map_lst)))
        #return cropped_img_lst
        return torch.Tensor(cropped_img_lst), torch.Tensor(np.array(cropped_map_lst))
        #return torch.Tensor(cropped_img_lst), torch.Tensor(cropped_map_lst), ann

class RandomCropAnGeneratePointMap(object):
    """
    """
    def __init__(self, width, height):
        """
        Args:
            width  (int): crop width
            height (int): crop height

        Returns:
            None
        """
        self.width = width
        self.height = height



    def _correctShape(self, img):
        a,b,c = img.shape
        if a == 3:
            img = img.transpose(1,2,0)
        return img

    def _correctImageType(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        return img

    def _randomCrop(self, img):
        seed = random.randint(0, 2147483647)
        random.seed(sedd)
        image_width, iamge_height = img.size
        rand_x = random.randint(0, width  - self.width  - 1)
        rand_y = random.randint(0, height - self.height - 1)

        roi = (rand_x, rand_y, rand_x + self.width, rand_y + self.height)
        return torch.Tensor(img.crop(roi)), rand_x, rand_y

    def _generateMap(self, roi, p_lst):
        # for raw image
        c_lst = []
        for p in p_lst:
            lst = [np.zeros((self.height, self.width))]
            if roi[0] <= p[0] and p[0] < roi[2] and roi[1] <= p[1] and p[1] < roi[2]:
                x_map, y_map = p[0] - roi[0], p[1] - roi[2]
                img = self.__generateGaussianMap(x_map, y_map, width, height)
                cnt += 1
                lst += [img]
            tmp = np.array(lst)
            c_lst += [tmp.max(0)]
        return torch.Tensor(np.array(c_lst))

    def __generateGaussianMap(self, x_0, y_0, width, height, sigma_x=100, sigma_y=100):
        y, x = np.mgrid[0:height, 0:width]
        return (255 * np.exp(-((x-x_0)**2 / sigma_x + (y - y_0)**2 / sigma_y))).astype(np.uint8)

    def __call__(self, img, p_lst, ann_lst):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """

        img = self._coorectImageType(self._correctShape(img))
        raw_img = img.copy()

        img = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale)

        cropped_img, roi = self._randomCrop(img)
        cropped_maps = self._generateMap(self, roi, p_lst)
        
        return cropped_img, maps

class MapGenerator(object):
    """
    """
    def __init__(self, sigma_x=100, sigma_y=100):
        """
        Args:
            width  (int): image width
            height (int): image height

        Returns:
            None
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def _generateMap(self, p_lst, width, height):
        # for raw image
        map_lst = []
        for p in p_lst:
            img = self.__generateGaussianMap(p[0], p[1], width, height, self.sigma_x, self.sigma_y)
            map_lst += [I.fromarray(img)]
        return map_lst

    def __generateGaussianMap(self, x_0, y_0, width, height, sigma_x=100, sigma_y=100):
        y, x = np.mgrid[0:height, 0:width]
        ret = (255 * np.exp(-((x-x_0)**2 / sigma_x + (y - y_0)**2 / sigma_y))).astype(np.uint8)
        return ret

    def __call__(self, img, p_lst):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        width, height = img.size

        map_lst = self._generateMap(p_lst, width, height)
        
        return map_lst
