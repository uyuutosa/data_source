import os

import cv2
import glob as g
import numpy as np
from numpy import array
import pandas as pd
import PIL.Image as I
from termcolor import colored, cprint
import s3fs
import pprint as pp

from . import transforms
from . import image_opener
mytransforms = transforms

class DataSouceIter:
    """Iterator for DataSource.

    Attributes:
        ds: A string the path contains data.
        train_test_split: The ratio of train and test. If 1 then no test data will be created.

    """
    def __init__(self, ds, **kwargs):
        self.ds = ds
        self.size = len(ds)
        self.i = 0

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        ret = self.ds[self.i]
        self.i += 1
        return ret

class BaseDataSource:
    """Base class for DataSource.

    Attributes:
        path: A string the path contains data.
        train_test_split: The ratio of train and test. If 1 then no test data will be created.
    """

    def __init__(self, path, train_test_ratio=1, **kwargs):
        self.path               = path
        self.train_test_ratio = train_test_ratio
        self.data_lst_dic     = self._getDataListDic(path)
        self.setMode("train")
        self.exif = mytransforms.CorrectExif()
        self.opener = image_opener.ImageOpener()

    def setDataListDic(self, path):
        self.data_lst_dic = self._getDataListDic(path)

    def getData(self, index, **kwargs):
        pass

    def _getDataListDic(self, image_path):
        train_lst = test_lst = []
        return {'train': train_lst, 'test': test_lst}

    def _loadImagePath(self, image_path):
        lst = []
        for root, dirs, files in os.walk(image_path):
            for file in files:
                names = file.split(".")
                if names[-1] == "JPG" or names[-1] == "jpg" or names[-1] == "png" or names[-1] == "json":
                    lst+=["/".join([root, file])]
        lst.sort()
        return lst

    def _trainTestSplit(self, lst):
        """Split the data source to 'train' and 'test'.
        A ratio of the number of data for 'train' and 'test' is decided by 'self.train_test_ratio'
        """

        train_lst = test_lst = []
        return train_lst, test_lst

    def setMode(self, mode):
        """Set mode of the data source. 'train' and 'test' modes are supported currently.

        Attributes:
            mode: a name of mode. please select 'train' or ' test'
        """

        self.mode = mode

    def __getitem__(self, index):
        return self.getData(index)

    def __len__(self):
        """Return a length for data source separated by 'train' and 'test'.
            If current mode was 'train' then the length for 'train' mode will be returned.
        """
        return len(self.data_lst_dic[self.mode])

    def __iter__(self):
        return DataSouceIter(self)

    def sendData(self):
        index = self.sock.recvData()
        data = self.getData(index)
        self.sock.sendData(data)

    def recvData(self, index):
        self.sock.sendData(index)
        data = self.sock.recvData()
        return data

    def sendInfo(self):
        size_dic = {}
        self.setMode('train')
        size_dic['train'] = self.__len__()
        self.setMode('test')
        size_dic['test']  = self.__len__()
        infos = {'size_dic'  : size_dic,
                 'mode'      : self.mode}
        self.sock.sendData(infos)

    def recvInfo(self):
        infos = self.sock.recvData()
        self.size_dic = infos['size_dic']
        self.mode     = infos['mode']

class DataSourceFromCam(BaseDataSource):
    """Data source from web camera. For implementation OpenCV is used internally.

    Attributes:
        path: A camera number.
        port: A port number for soccet connection.
        train_test_split: The ratio of train and test. If 1 then no test data will be created.

    """

    def __init__(self, path, train_test_ratio=1, **kwargs):
        super(DataSourceFromCam, self).__init__(path, train_test_ratio)
        self.cap = cv2.VideoCapture(path)

    def _getDataListDic(self, image_path):
        train_lst = test_lst = []
        return {'train': train_lst, 'test': test_lst}

    def getData(self, index, **kwargs):
        ret, frame = self.cap.read()
        return I.fromarray(frame[:,:,::-1])

    def __len__(self):
        return 1000000000 # length cannot be evaluate

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

class DataSourceFromDirContainedFiles(BaseDataSource):
    """Data source from a nested directory contained images.
    Attributes:
        path: A string the path contains data. the directry structure is below:
             path - 001_images - a.jpg
                   |                 |- b.jpg
                   |                 :
                   |
                   |- 002_images - a.jpg
                   |                  |- b.jpg
                   |                 :
                   :

        train_test_split: The ratio of train and test. If 1 then no test data will be created.

    """

    def __init__(self, path, train_test_ratio=1, indices=None, is_random_pick=False, is_ret_idx=False, **kwargs):
        self.is_random_pick = is_random_pick
        self.is_ret_idx = is_ret_idx
        if indices is None:
            self.indices = None
        #    self.indices = range(len(g.glob(os.path.join(g.glob(os.path.join(path, '*/'))[0], '*'))))
        elif type(indices) == int:
            self.indices = [int(i) for i in indices.split(',')]
        super(DataSourceFromDirContainedFiles, self).__init__(path, train_test_ratio)

    def _getDataListDic(self, image_path):
        train_lst, test_lst =\
            self._trainTestSplit(self._loadImagePath(image_path))

        return {'train': train_lst, 'test': test_lst}

    def _loadImagePath(self, image_path):
        # return directory list
        lst = g.glob(os.path.join(image_path, "*/"))
        lst.sort()
        return lst

    def _trainTestSplit(self, lst):
        """Split the data source to 'train' and 'test'.
        A ratio of the number of data for 'train' and 'test' is decided by 'self.train_test_ratio'
        """

        size                      = len(lst)
        train_size                = int(size*self.train_test_ratio)
        train_lst, test_lst = lst[0:train_size], lst[train_size:]
        return train_lst, test_lst

    def _getDataSouce(self, index, sub_idx=None):
        path = os.path.join(self.data_lst_dic[self.mode][index])
        lst = g.glob(os.path.join(path, "*"))
        
        if os.path.isdir(lst[0]):
            ds = DataSourceFromDirContainedFiles(path, train_test_ratio=1, is_random_pick=self.is_random_pick)
            if sub_idx is not None:
                idx = sub_idx
            else:
                idx = np.random.randint(len(ds))
            #pp.pprint(ds.data_lst_dic["train"])
            lst = ds.getDataPath(idx)
        #print(lst[0])
            #input(lst)
        
        return lst

    def getData(self, index, sub_idx=None, **kwargs):
        #print(index, sub_idx)
        path_lst = self._getDataSouce(index, sub_idx)
        path_lst.sort()
        self.pl = path_lst
        if self.indices is not None:
            tmp = []
            for index in self.indices:
                tmp += [path_lst[index]]
            path_lst = tmp
            path_lst.sort()
        lst = [self.exif(self.opener.open(x)) for x in path_lst]
        if self.is_random_pick:
            if sub_idx is not None:
                idx = sub_idx
            else:
                idx = np.random.randint(len(lst))
            ret = lst[idx]
            if self.is_ret_idx:
                return ret, idx
            else:
                return ret
        else:
            return lst

    def getDataPath(self, index):
        path_lst = g.glob(os.path.join(self.data_lst_dic[self.mode][index],"*"))
        path_lst.sort()
        if self.indices is not None:
            tmp = []
            for index in self.indices:
                tmp += [path_lst[index]]
            path_lst = tmp
            path_lst.sort()
        return path_lst

class DummyDataSource(BaseDataSource):
    """Data source from a directory contained images.

    Attributes:
        path: A string the path contains data. the directry structure is below:
             path - 001_image.jpg
                   |- 002_image.jpg
                   :

        train_test_split: The ratio of train and test. If 1 then no test data will be created.

    """
    def __init__(self, path,  train_test_ratio=1, size=0, **kwargs):
        super(DummyDataSource, self).__init__(path, train_test_ratio)
        self.size = size

    def _getDataListDic(self, image_path):
        return {'train': ['dummy'] * int(self.size * train_test_ratio),
                'test':  ['dummy'] * int(self.size * (1 - train_test_ratio))}

    def _trainTestSplit(self, lst):
        """Split the data source to 'train' and 'test'.
        A ratio of the number of data for 'train' and 'test' is decided by 'self.train_test_ratio'
        """

        size                      = len(lst)
        train_size                = int(size*self.train_test_ratio)
        train_lst, test_lst = lst[0:train_size], lst[train_size:]
        return train_lst, test_lst

    def getData(self, index, **kwargs):
        return self.data_lst_dic[self.mode][index]


class DataSourceFromFiles(BaseDataSource):
    """Data source from a directory contained images.

    Attributes:
        path: A string the path contains data. the directry structure is below:
             path - 001_image.jpg
                   |- 002_image.jpg
                   :

        train_test_split: The ratio of train and test. If 1 then no test data will be created.

    """
    def __init__(self, path,  train_test_ratio=1, **kwargs):
        super(DataSourceFromFiles, self).__init__(path, train_test_ratio)

    def _getDataListDic(self, image_path):
        train_lst, test_lst =\
            self._trainTestSplit(self._loadImagePath(image_path))

        return {'train': train_lst, 'test': test_lst}

    def _trainTestSplit(self, lst):
        """Split the data source to 'train' and 'test'.
        A ratio of the number of data for 'train' and 'test' is decided by 'self.train_test_ratio'
        """

        size                      = len(lst)
        train_size                = int(size*self.train_test_ratio)
        train_lst, test_lst = lst[0:train_size], lst[train_size:]
        return train_lst, test_lst

    def getData(self, index, **kwargs):
        return self.exif(self.opener.open(self.data_lst_dic[self.mode][index]))

    def getDataPath(self, index, **kwargs):
        return self.data_lst_dic[self.mode][index]

class DataSourceFromList(BaseDataSource):
    """Data source contains image data from list.

    Attributes:
        lst: A list contains image paths.
        train_test_split: The ratio of train and test. If 1 then no test data will be created.
    """

    def __init__(self, lst, train_test_ratio=1, **kwargs):
        super(DataSourceFromList, self).__init__(lst,  train_test_ratio)

    def _getDataListDic(self, lst):
        train_lst, test_lst =\
            self._trainTestSplit(lst)

        return {'train': train_lst, 'test': test_lst}

    def _trainTestSplit(self, lst):
        """Split the data source to 'train' and 'test'.
        A ratio of the number of data for 'train' and 'test' is decided by 'self.train_test_ratio'
        """
        size                = len(lst)
        train_size          = int(size*self.train_test_ratio)
        train_lst, test_lst = lst[0:train_size], lst[train_size:]
        return train_lst, test_lst

    def getData(self, index, **kwargs):
        return self.exif(self.opener.open(self.data_lst_dic[self.mode][index]))

    def getDataPath(self, index, **kwargs):
        return self.data_lst_dic[self.mode][index]

class DataSourceFromImage(BaseDataSource):
    """Data source contains only one image.

    Attributes:
        path: A image path.
        train_test_split: The ratio of train and test. If 1 then no test data will be created.
    """

    def __init__(self, path, train_test_ratio=1, **kwargs):
        super(DataSourceFromImage, self).__init__(path, train_test_ratio)

    def _getDataListDic(self, image_path):
        return {'train': [image_path], 'test': [image_path]}

    def getData(self, index, **kwargs):
        return self.exif(self.opener.open(self.data_lst_dic[self.mode][index]))

class DataSourceFromNumpy(BaseDataSource):
    """Data source contains only one image.

    Attributes:
        path: A image path.
        train_test_split: The ratio of train and test. If 1 then no test data will be created.
    """

    def __init__(self, path, train_test_ratio=1, **kwargs):
        super(DataSourceFromImage, self).__init__(path, train_test_ratio)

    def _getDataListDic(self, npy_path):
        return {'train': [npy_path], 'test': [npy_path]}

    def getData(self, index, **kwargs):
        return np.loads(self.data_lst_dic[self.mode][index])


class DataSourceFromMovie(BaseDataSource):
    """Data source contains image data from a movie file.

    Attributes:
        path: A path for a movie.
        train_test_split: The ratio of train and test. If 1 then no test data will be created.
    """

    def __init__(self, path,  train_test_ratio=1, **kwargs):
        super(DataSourceFromMovie, self).__init__(path, train_test_ratio)

    def _getDataListDic(self, image_path):
        self.movie_path = image_path
        self.cap  = cv2.VideoCapture(self.movie_path)
        if not self.cap.isOpened():
            wrong = colored('wrong', 'red')
            raise ValueError("{} is {} movie path".format(self.movie_path, wrong))

        self.fps  = self.cap.get(cv2.CAP_PROP_FPS)

        # define total size of movie (minus offsets)
        self.size = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret_val =  self._trainTestSplit(
                   image_path)

        self.cap.release()
        return ret_val

    def _trainTestSplit(self, lst):
        """Split the data source to 'train' and 'test'.
        A ratio of the number of data for 'train' and 'test' is decided by 'self.train_test_ratio'
        """
        # define train and test  size
        train_size = int(self.size*self.train_test_ratio)
        test_size  = self.size - train_size

        self.size_dic = {'train': train_size,
                         'test' : test_size}

        offset_dic    = {'train': 0,
                         'test' : train_size}

        return offset_dic

    def getData(self, index):
        self.cap  = cv2.VideoCapture(self.movie_path)
        if not self.cap.isOpened():
            wrong = colored('wrong', 'red')
            raise ValueError("{} is {} movie path".format(self.movie_path, wrong))
        # data list dic behave offset dic
        try:
#            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(index + self.data_lst_dic[self.mode]))
            self.pre_frame = frame = I.fromarray(self.cap.read()[1][...,::-1])
            self.cap.release()
            return frame
        except:
            unknown = colored('UnKnownError:')
            print("\n{} currently under inspection, skip.".format(unknown))
            self.cap.release()
            try:
                return self.pre_frame
            except:
                return I.fromarray(np.ones((10, 10, 3)).astype(np.uint8))



    def __len__(self):
        return self.size



class DataSourceFromCSV(BaseDataSource):
    """Data source contains an assigned array data from a CSV file.

    Attributes:
        path: A path for a CSV file.
        train_test_split: A ratio of train and test. If 1 then no test data will be created.
        column_lst      : A list contains names of column names for CSV.
    """

    def __init__(self,
                     path,
                     train_test_ratio=1,
                     column_lst=None,
                     data_format=None,
                     **kwargs):

        self.column_lst = column_lst
        self.data_format = data_format
        super(DataSourceFromCSV, self).__init__(path, train_test_ratio)

    def _extract_concat(self, df, lst):
        concat_lst = []
        for name in lst:
            concat_lst += [df[name]]
        return pd.concat(concat_lst, axis=1)

    def setColumnList(self, column_lst):
        self.column_lst = column_lst
        self.data_lst_dic = self._getDataListDic(self.path)

    def _getDataListDic(self, path):
        df = pd.read_csv(path)
        if self.column_lst is not None:
            df = self._extract_concat(df, self.column_lst)
        train_lst, test_lst = self._trainTestSplit(df)
        return {'train': train_lst, 'test': test_lst}

    def _loadCSV(self, csv_path):
        df = pd.read_csv(csv_path)

    def _trainTestSplit(self, lst):
        """Split the data source to 'train' and 'test'.
        A ratio of the number of data for 'train' and 'test' is decided by 'self.train_test_ratio'
        """
        size                = len(lst)
        train_size          = int(size*self.train_test_ratio)
        train_lst, test_lst = lst.iloc[0:train_size], lst.iloc[train_size:]
        return train_lst, test_lst

    def getData(self, index, **kwargs):
        if self.data_format == 'pandas':
            return self.data_lst_dic[self.mode].iloc[index]
        else:
            return array(self.data_lst_dic[self.mode])[index]

    def __getitem__(self, index):
        return self.getData(index)


class DataSourceFromMultipleSouces(BaseDataSource):
    """Source contains multiple data and format

    Attributes:
        path: A string the path contains data.
        train_test_split: The ratio of train and test. If 1 then no test data will be created.
        kwag:              Misc. for child class of data souce.

    """

    def __init__(self, path, train_test_ratio=1, **kwargs):
        self.kwargs = kwargs
        super(DataSourceFromMultipleSouces, self).__init__(path, train_test_ratio)


    def _getDataListDic(self, image_path):
        """Get data list dictionary has image paths separated of 'train' and 'test'

        Attributes:
            image_path: A path list contains sevaral image path.

        """
        #import  aimaker.data.data_source_factory as dsf
        from . import data_source_factory
        dsf = data_source_factory

        factory = dsf.DataSourceFactory(self.train_test_ratio)
        data_source_lst = []
        offset_lst_train = []
        offset_lst_test  = []

        for path in image_path.split(","):
            ds = factory.create(path, **self.kwargs)
            mode = ds.mode
            data_source_lst += [ds]
            ds.setMode("train")
            offset_lst_train += [len(ds)]
            ds.setMode("test")
            offset_lst_test  += [len(ds)]
            ds.setMode(mode)

        self.data_source_lst = data_source_lst
        data_lst_dic = {}
        # mimic offset list as data path list
        data_lst_dic["train"] = np.cumsum(offset_lst_train).tolist()
        data_lst_dic["test"]  = np.cumsum(offset_lst_test).tolist()

        return data_lst_dic


    def getData(self, index, **kwargs):
        """Get data from the data source.

        Get data from the data source separated of 'train and 'test'.
        If current data mode is 'train' then getting data is extracted from the list for 'train'.
        If you want to get data from 'test' deta, please change the mode to 'test' by 'self.setMode()'

        Attributes:
            index: An index for the image number.

        """
        pre_offset = 0
        size = len(self)
        for i, offset in enumerate(self.data_lst_dic[self.mode]):
            if offset <= index:
                if size != index:
                    pre_offset = offset
                else:
                    break
            else:
                break

        return self.data_source_lst[i].getData(index-pre_offset, **kwargs)

    def getDataPath(self, index, **kwargs):
        """Get data from the data source.

        Get data from the data source separated of 'train and 'test'.
        If current data mode is 'train' then getting data is extracted from the list for 'train'.
        If you want to get data from 'test' deta, please change the mode to 'test' by 'self.setMode()'

        Attributes:
            index: An index for the image number.

        """
        pre_offset = 0
        size = len(self)
        for i, offset in enumerate(self.data_lst_dic[self.mode]):
            if offset <= index:
                if size != index:
                    pre_offset = offset
                else:
                    break
            else:
                break

        return self.data_source_lst[i].getDataPath(index-pre_offset, **kwargs)

    def setMode(self, mode):
        """Set mode of the data source. 'train' and 'test' modes are supported currently.

        Attributes:
            mode: a name of mode. please select 'train' or ' test'
        """

        self.mode = mode
        for ds in self.data_source_lst:
            ds.setMode(mode)

    def __len__(self):
        """Return a length for data source separated by 'train' and 'test'.
            If current mode was 'train' then the length for 'train' mode will be returned.
        """

        total_size = 0
        for ds in self.data_source_lst:
            total_size += len(ds)

        return total_size

