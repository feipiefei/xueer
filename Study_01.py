import numpy as np
import os
from PIL import Image
import random
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.externals import joblib              //导入支援库


def img2vec(fname):
    '''将jpg等格式的图片转为向量'''
    im = Image.open(fname).convert('L')
    im = im.resize((28,28))
    tmp = np.array(im)
    vec = tmp.ravel()
    return vec

def split_data(paths):
    '''随机抽取1000张图片作为训练集'''
    fn_list = os.llistdir(paths)
    X = []
    y = []
    d0 = random.sample(fn_list,1000)
    for i,name in enumerate(d0):
        y.append(name[0])
        X.append(img2vec(name))
        dataset = np.array([X,y])
    return X,y

def knn_clf(X_train,label):
    '''构建分类器'''
    clf = knn()
    clf.fit(X_train,label)
    return clf

def save_model(model,output_name):
    '''保存模型'''
    joblib.dump(model,ouotput_name)
    X_train, y_label = split_data(file_path)
    clf = knn_clf(X_train, y_label)
    save_model(clf, 'mnist_knn1000.m')

    class MainWindow(wx.Frame):
        def __init__(self, parent, title):
            wx.Frame.__init__(self, parent, title=title, size=(600, -1))
            static_font = wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL)

            Size = namedtuple("Size", ['x', 'y'])
            s = Size(100, 50)
            sm = Size(100, 25)

            self.fileName = None
            self.model = model

            b_labels = [u'open', u'run']

            TipString = [u'选择图片', u'识别数字']

            funcs = [self.choose_file, self.run]

            '''create input area'''
            self.in1 = wx.TextCtrl(self, -1, size=(2 * s.x, 3 * s.y))
            self.out1 = wx.TextCtrl(self, -1, size=(s.x, 3 * s.y))

            '''create button'''
            self.sizer0 = wx.FlexGridSizer(rows=1, hgap=4, vgap=2)
            self.sizer0.Add(self.in1)

            buttons = []
            for i, label in enumerate(b_labels):
                b = wx.Button(self, id=i, label=label, size=(1.5 * s.x, s.y))
                buttons.append(b)
                self.sizer0.Add(b)

            self.sizer0.Add(self.out1)

            '''set the color and size of labels and buttons'''
            for i, button in enumerate(buttons):
                button.SetForegroundColour('red')
                button.SetFont(static_font)
                button.SetToolTipString(TipString[i])
                button.Bind(wx.EVT_BUTTON, funcs[i])

            '''layout'''
            self.SetSizer(self.sizer0)
            self.SetAutoLayout(1)
            self.sizer0.Fit(self)

            self.CreateStatusBar()
            self.Show(True)


def run(self, evt):
    if self.fileName is None:
        self.raise_msg(u'请选择一幅图片')
        return None
    else:
        model_path = os.path.join(origin_path, 'mnist_knn1000.m')
        clf = model.load_model(model_path)
        ans = model.tester(self.fileName, clf)
        self.out1.Clear()
        self.out1.write(str(ans))


def choose_file(self, evt):
    '''choose img'''
    dlg = wx.FileDialog(
        self, message="Choose a file",
        defaultDir=os.getcwd(),
        defaultFile="",
        wildcard=wildcard,
        style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
    )
    if dlg.ShowModal() == wx.ID_OK:
        paths = dlg.GetPaths()
        dlg.Destroy()
        self.in1.Clear()
        self.in1.write(paths[0])
        self.fileName = paths[0]
        im = Image.open(self.fileName)
        im.show()
    else:
        return None