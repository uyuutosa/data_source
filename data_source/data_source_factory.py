from .data_source import *
import glob as g
import os


class DataSourceFactory():
    def __init__(self,  train_test_ratio=0.5):

        self.data_dic = {   'file'    : DataSourceFromFiles,
                            'image'   : DataSourceFromImage,
                            'movie'   : DataSourceFromMovie,
                            'multi'   : DataSourceFromMultipleSouces,
                            'cam'     : DataSourceFromCam,
                            'lst'     : DataSourceFromList,
                            'csv'     : DataSourceFromCSV,
                #            'numpy'   : DataSourceFromNumpy,
                            'dir'     : DataSourceFromDirContainedFiles,
                            'dummy'   : DummyDataSource
                             }

        self.support_movie_ext_lst = ["mp4", "avi", 'mov']
        self.support_csv_ext_lst   = ["csv", "txt"]
        self.train_test_ratio = train_test_ratio

    def _parseExtension(self, path):
        if path == 'dummy':
            return path, path
        if isinstance(path, list):
            return "lst", path
        if isinstance(path, int):
            return 'cam', path
        elif ',' in path:
            return "multi", path
        elif path.split('.')[-1] in self.support_csv_ext_lst:
            return 'csv', path
        else:
            file_or_movie = path.split('/')[-1]
            if "." in file_or_movie:
                ext = file_or_movie.split(".")[-1]
                if ext in self.support_movie_ext_lst:
                    return "movie", path
                else:
                    raise "{} is unsupported extension.".format(ext)
            else:
                if self._isFileExists(path):
                    for ext in self.support_movie_ext_lst:
                        lst = g.glob(path + "/*." + ext)
                        if len(lst): 
                            return 'multi', ",".join(lst)
                        return  "file", path
                else:
                    return  "dir", path

    def _isFileExists(self, path, ext="*"):
        lst = g.glob(os.path.join(path, "*." + ext))
        if len(lst):
            return True
        else:
            False

        
    def create(self, path, **kwargs):
        name, path = self._parseExtension(path)
        if name == 'multi':
            pass
        elif 'any_data_source' in kwargs:
            name = 'any_data_source'
            self.data_dic[name] = kwargs[name]
            del kwargs[name]

        if not name in self.data_dic:
            raise NotImplementedError(('{} is wrong key word for ' + \
                                       '{}. choose {}')\
                                      .format(name, self.__class__.__name__, self.data_dic.keys()))

        return self.data_dic[name](path,  self.train_test_ratio, **kwargs)
