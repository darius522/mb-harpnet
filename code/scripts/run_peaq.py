import matlab.engine
import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import soundfile as sf
from scipy.signal import resample_poly

def format_and_write(path, tmp_path):
    a, sr = sf.read(path, always_2d=True)
    a = resample_poly(a, 480, sr//100)
    a = np.repeat(a, 2, 1)
    sf.write(tmp_path, a, 48000)

def twof_model(AvgModDiff1, ADB, clip=True):
    '''
    The 2f-model is given by the following euqation:
        MMSest = 56.1345 / (1 + (−0.0282 · AvgModDiff1 − 0.8628)^2) − 27.1451 · ADB + 86.3515
    '''
    mms_est = (56.1345 / (1 + (-0.0282 * AvgModDiff1 - 0.8628)**2)) - 27.1451 * ADB + 86.3515
    return np.clip(mms_est, 0.0, 100.0) if clip else mms_est

def get_parser():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str,
         default='/mm1/petermann/datasets/dnr_v2_hq',
         help='Root directory of DNR')
    parser.add_argument('--pred-dir', type=str,
         default='/mm1/petermann/tums/experiments/dnr_v2/high_res/masktcn_174427/audio_output',
         help='Prediction directory of DnR')
    parser.add_argument('--source', type=str,
         default='sfx',
         help='DNR task labels. Could be music, sfx, or speech')
    parser.add_argument('--alias', type=str,
         default='src1',
         help='The source alias used by predictor (usually src0, 1, 2')
    parser.add_argument('--noisy', type=int,
         default=0,
         help='Whether or not to use full mix as prediction (low anchor)')
    parser.add_argument('--oracle', type=int,
         default=0,
         help='Whether or not to use ground truth as prediction (high anchor)')
    parser.add_argument('--tag', type=str,
         default='None',
         help='Added to csv name')
    return parser

def main(args):
    pass
    # path_to_peaq_toolbox = "/mm1/petermann/dnr_remix/PQevalAudio-v1r0"
    # df = pd.DataFrame(columns=['ref_file', 'est_file', 'AvgModDiff1' , 'ADB', 'MMS'])
    
    # afiles = os.listdir(os.path.join(args.data_dir,'tt'))
    # afiles = [x for x in afiles if os.path.isdir(os.path.join(os.path.join(args.data_dir,'tt',x)))]

    # m = matlab.engine.start_matlab()
    # m.eval("addpath(genpath('{}'));".format(path_to_peaq_toolbox))

    # pbar = tqdm(afiles)
    # for f in pbar:
    #     pbar.set_description("Processing %s" % f)

    #     # 1. Get ref+est
    #     ref = os.path.join(args.data_dir,'tt',f,args.source+'.wav')
    #     if bool(args.noisy):
    #         est = os.path.join(args.data_dir,'tt',f,'mix.wav')
    #     elif bool(args.oracle):
    #         est = os.path.join(args.data_dir,'tt',f,args.source+'.wav')
    #     else:
    #         est = os.path.join(args.pred_dir,ref.split('/')[-2]+'_'+args.alias+'.wav')
            
    #     # 2. Resample + stereo convert (create tmp file)
    #     ref_path = './tmp/ref_tag={}_source={}.wav'.\
    #                         format(args.tag, args.source)
    #     est_path = './tmp/est_tag={}_source={}.wav'.\
    #                         format(args.tag, args.source)                 
    #     format_and_write(ref, tmp_path=ref_path)
    #     format_and_write(est, tmp_path=est_path)  
        
    #     # 3. Compute PEAQ
    #     try:
    #         results = m.PQevalAudio([ref_path], [est_path])
    #         '''
    #         MOVs is structured as follows:
    #             [0]:BandwidthRefB, [1]:BandwidthTestB, [2]:Total NMRB, [3]:WinModDiff1B, 
    #             [4]:ADBB, [5]:EHSB, [6]:AvgModDiff1B, [7]:AvgModDiff2B, [8]:RmsNoiseLoudB, 
    #             [9]:MFPDB, [10]:RelDistFramesB 
    #         '''
            
    #         # 4. Get 2f-model output
    #         MOV = np.array(results['MOVB'][0])
    #         AvgModDiff1, ADB = MOV[6], MOV[4]
    #         mms_est = twof_model(AvgModDiff1, ADB)
    #     except:
    #         AvgModDiff1, ADB, mms_est = np.nan, np.nan, np.nan
        
    #     # 5. Write to file
    #     df = df.append({'ref_file':ref, 'est_file':est, 'AvgModDiff1':AvgModDiff1,'ADB':ADB, 'MMS':mms_est}, ignore_index=True)
    
    # df.to_csv(os.path.join('/mm1/petermann/cfp_sed/PQevalAudio-v1r0/output/peaq/models_no_boundaries','peaq_tag={}_source={}.csv').\
    #                         format(args.tag, args.source))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)