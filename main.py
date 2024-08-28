import numpy as np
import matplotlib.pyplot as plt
#import CMIP6_second_GFDL_atlantic

skill_atlantic=np.zeros((9))
auto_atlantic=np.zeros((98))

#import CMIP6_second_GFDL_atlantic
skill_pacific=np.zeros((9))
auto_pacific=np.zeros((98))

skill_indian=np.zeros((9))
auto_indian=np.zeros((98))

skill_so=np.zeros((9))
auto_so=np.zeros((98))

 
exec(open("final_code_no3_with_reg2.py").read())
exec(open("final_model_code.py").read())
exec(open("final_code_TCNN_IG_SHAP.py").read())

