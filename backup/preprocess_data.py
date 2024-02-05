import os
from PIL import Image
import pandas as pd
from tqdm import tqdm


class DataHandler():
    def __init__(self, opt):
        self.opt = opt
        #self.create_records()

    def create_records(self):
        split_path = self.opt['dataset']['ann_path']
        # get all the file names in the split path
        split_files = os.listdir(split_path)
        class_names = {
            'absent_septum' : 0,
            'artefacts' : 0,
            'craniatomy' : 0,
            'dural' : 0,
            'ea_mass' : 0,
            'edema' : 0,
            'encephalomalacia' : 0,
            'enlarged_ventricles' : 0,
            'intraventricular' : 0,
            'lesions' : 0,
            'local' : 0,
            'mass' : 0,
            'mass_all' : 0,
            'other' : 0,
            'posttreatment' : 0,
            'resection' : 0,
            'sinus' : 0,
            'wml' : 0
        }

        # create a pandas dataframe with the column filename, type, mask_pos, mask_neg, split
        data_df = pd.DataFrame(columns=['filename', 'type', 'full_map', 'mask_pos', 'mask_neg', 'split'])


        for file in split_files:
            file_path = os.path.join(split_path, file)
            df = pd.read_csv(file_path)
            if 'ixi' in file:
                path = self.opt['dataset']['path']
                df['filename'] = df['filename'].str.replace('./data', path, regex=False)
                filenames = df['filename'].tolist()
                num_filenames = len(filenames)
                # create a type list with the length of filenames and append normal
                type_list = ['normal_ixi'] * num_filenames
                full_map_list = [None] * num_filenames
                mask_pos_list = [None] * num_filenames
                mask_neg_list = [None] * num_filenames
                split_list = ['train'] * num_filenames

                # apextend the dataframe with the new data
                data_to_append = pd.DataFrame({
                    'filename': filenames,
                    'type': type_list,
                    'full_map': full_map_list,
                    'mask_pos': mask_pos_list,
                    'mask_neg': mask_neg_list,
                    'split': split_list
                })
                # Use concat instead of append
                data_df = pd.concat([data_df, data_to_append], ignore_index=True)



            else:

                path = self.opt['dataset']['path']
                if 'normal' in file:
                    if file == 'normal_test_ann.csv':
                        continue

                    else:
                        df['filename'] = df['filename'].str.replace('./data', path, regex=False)
                        #df['full_map'] = df['filename']
                        #df['full_map'] = df['full_map'].str.replace('.png', '_brain_map_full.png', regex=False)

                        filenames = df['filename'].tolist()
                        type_list = ['normal_fast'] * len(filenames)
                        full_map_list = [None] * len(filenames)

                        num_filenames = len(filenames)
                        if file == 'normal_train.csv':
                            split_list = ['train'] * num_filenames
                        elif file == 'normal_val.csv':
                            split_list = ['val'] * num_filenames
                        elif file == 'normal_test.csv':
                            split_list = ['test'] * num_filenames

                        if file == 'normal_train.csv' or file == 'normal_val.csv':
                            mask_pos_list = [None] * num_filenames
                            mask_neg_list = [None] * num_filenames
                        else:
                            mask_pos_file = os.path.join(split_path, 'normal_test_ann.csv')
                            mask_pos_df = pd.read_csv(mask_pos_file)
                            mask_pos_df['filename'] = mask_pos_df['filename'].str.replace('./data', path, regex=False)
                            mask_pos_list = mask_pos_df['filename'].tolist()
                            mask_neg_list = [None] * len(mask_pos_list)

                        data_to_append = pd.DataFrame({
                            'filename': filenames,
                            'type': type_list,
                            'full_map': full_map_list,
                            'mask_pos': mask_pos_list,
                            'mask_neg': mask_neg_list,
                            'split': split_list
                        })

                        # Use concat instead of append
                        data_df = pd.concat([data_df, data_to_append], ignore_index=True)

                else:
                    abnormal_class = file.split('.csv')[0]
                    abnormal_class = abnormal_class.split('_ann')[0]
                    abnormal_class = abnormal_class.split('_neg')[0]

                    if class_names[abnormal_class] == 0:
                        class_names[abnormal_class] = 1
                        img_file = os.path.join(split_path, (abnormal_class + '.csv'))
                        img_neg_file = os.path.join(split_path, (abnormal_class + '_neg.csv'))
                        img_pos_file = os.path.join(split_path, (abnormal_class + '_ann.csv'))

                        filenames = []
                        mask_neg_list = []
                        mask_pos_list = []
                        type_list = []
                        split_list = []
                        length = 0
                        # check if the file exists
                        if os.path.isfile(img_file):
                            img_df = pd.read_csv(img_file)
                            img_df['filename'] = img_df['filename'].str.replace('./data', path, regex=False)
                            img_df['full_map'] = img_df['filename']
                            img_df['full_map'] = img_df['full_map'].str.replace('.png', '_brain_map_full.png', regex=False)


                            filenames = img_df['filename'].tolist()
                            full_map_list = img_df['full_map'].tolist()
                            
                            num_filenames = len(filenames)
                            type_list = [abnormal_class] * num_filenames
                            split_list = ['test'] * num_filenames
                            length = num_filenames

                        if os.path.isfile(img_neg_file):
                            img_neg_df = pd.read_csv(img_neg_file)
                            img_neg_df['filename'] = img_neg_df['filename'].str.replace('./data', path, regex=False)
                            mask_neg_list = img_neg_df['filename'].tolist()


                        if os.path.isfile(img_pos_file):
                            img_pos_df = pd.read_csv(img_pos_file)
                            img_pos_df['filename'] = img_pos_df['filename'].str.replace('./data', path, regex=False)
                            mask_pos_list = img_pos_df['filename'].tolist()

                        if mask_neg_list == []:
                            mask_neg_list = [None] * length
                        if mask_pos_list == []:
                            mask_pos_list = [None] * length
                        if type_list == []:
                            type_list = [None] * length
                        if split_list == []:
                            split_list = [None] * length


                        data_to_append = pd.DataFrame({
                            'filename': filenames,
                            'type': type_list,
                            'full_map': full_map_list,
                            'mask_pos': mask_pos_list,
                            'mask_neg': mask_neg_list,
                            'split': split_list
                        })
                        # Use concat instead of append
                        data_df = pd.concat([data_df, data_to_append], ignore_index=True)

                    else:
                        continue



        return data_df

