import sys
import yaml

# corpus的路径必须用os.sep
# spellInputSize是字符表长度
from exe_model.dual_model import LipReading

if __name__ == '__main__':

    argvLen = len(sys.argv)
    if argvLen < 2:
        print(f'lack of modelType')
    else:
        modelType = sys.argv[1]
        yamlPath = 'source/config/model.yaml'
        file = open(yamlPath, 'r')
        paras = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        paras['mode'] = modelType

        if argvLen == 3:
            paras['checkPath'] = sys.argv[2]

        lipModel = LipReading(paras)
        lipModel.Train(0)