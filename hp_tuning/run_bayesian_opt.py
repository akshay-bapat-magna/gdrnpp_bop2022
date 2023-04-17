from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import subprocess
from datetime import datetime

iter_count = 0

def even(num):
    if num % 2 > 0:
        return num + 1
    else:
        return num

def blackbox_func(lr_r, bs_r, m):
    global iter_count
    iter_count += 1
    lr = float(10**(-lr_r))
    # betas = (float(m), 0.999)
    betas = float(m)
    bs = even(int(2*bs_r))
    
    (f"Starting Bayesian Optimization Iteration with LR_r: {lr_r}, BS_r:{bs_r}, momentum: {m}")
    subprocess.call([
        './core/gdrn_modeling/train_gdrn_bo.sh',
        './configs/gdrn/doorlatch/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_doorlatch.py',
        '0,1',
        f'{lr}',
        f'{bs}',
        f'{betas}',
        f'{iter_count}',
    ])

    output_file = "./output/gdrn/doorlatch/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_doorlatch/inference_/" + \
    "doorlatch_bop_test_pbr/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-doorlatch_doorlatch_bop_test_pbr_tab.txt"
    run_info = f"LR: {10**(-lr_r)}, BS: {bs}, momentum: {m}"
    
    with open(output_file, 'r') as f:
        for line in f.readlines():
            l = line.split()
            if l[0] == 'ad_5':
                retval = float(l[2])
    
    with open("bayesian_opt_log.txt", "a") as f:
        f.write(f"\n{run_info} --- ADD5: {retval}")
    
    return retval



pbounds = {
    'lr_r': (1, 7),
    'bs_r': (1, 64),
    'm': (0.85, 0.95)
}

load = True

if load:
    opt = BayesianOptimization(
        f=blackbox_func,
        pbounds=pbounds,
    )

    load_logs(opt, logs=["./hp_tuning/logs.json"])
    logger = JSONLogger(path="./hp_tuning/logs.json", reset=False)
    opt.subscribe(Events.OPTIMIZATION_STEP, logger)

    opt.maximize(
        init_points=40,
        n_iter=50,
    )

    for i, res in enumerate(opt.res):
        print(f"Iteration {i}: \n\t{res}")

    print("\n\nMaximized:\n", opt.max)
    
else:
    with open("bayesian_opt_log.txt", "a") as f:
        line = "--"*40
        f.write(f"\n{line}\nBayesian Optimization started: {datetime.now()}\n")

    opt = BayesianOptimization(
        f=blackbox_func,
        pbounds=pbounds,
    )

    logger = JSONLogger(path="./hp_tuning/logs.json", reset=True)
    opt.subscribe(Events.OPTIMIZATION_STEP, logger)

    opt.maximize(
        init_points=40,
        n_iter=50,
    )

    for i, res in enumerate(opt.res):
        print(f"Iteration {i}: \n\t{res}")

    print("\n\nMaximized:\n", opt.max)