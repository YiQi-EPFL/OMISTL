import os
import pickle


def store_infeasible(file,dataset_name):
    file_s = file
    relative_path = os.getcwd()
    if not os.path.isdir(os.path.join(relative_path, 'data', dataset_name)):
        os.mkdir(os.path.join(relative_path + '/data/' + dataset_name + 'infeasible' + '.p'))
    file_path = os.path.join(relative_path, 'data', dataset_name,'infeasible'+'.p')
    outfile = open(file_path, "wb")
    pickle.dump(file_s, outfile);
    outfile.close()
    return file_path
