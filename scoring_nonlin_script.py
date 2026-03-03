# Scoring nonlin script

import glob, os
import numpy as np
from clean_utils import *

def loss_pdf_cdf(target, synth, num_points=1000):
    """Compute the loss of the PDF using the CDF (integration)"""
    t_flat = target.flatten()
    s_flat = synth.flatten()
    
    # Define a common support for evaluation
    min_val = min(t_flat.min(), s_flat.min())
    max_val = max(t_flat.max(), s_flat.max())
    x = np.linspace(min_val, max_val, num_points)
    
    #Compute empirical CDF
    cdf_target = np.searchsorted(np.sort(t_flat), x, side='right') / len(t_flat)
    cdf_synth = np.searchsorted(np.sort(s_flat), x, side='right') / len(s_flat)
    
    # Integrate
    return np.trapz(np.abs(cdf_target - cdf_synth), x)

def loss_ps_log(target, synth):
    """Computes the power spectrum term of the loss in log-log scale"""
    _, ps_target = power_spectrum(target)
    _, ps_synth = power_spectrum(synth)
    
#add an offset to prevent the divergence of the log
    eps = 1e-10
    log_ps_t = np.log10(ps_target + eps)
    log_ps_s = np.log10(ps_synth + eps)
    
    # Mean squarred error
    return np.mean((log_ps_t - log_ps_s)**2)

def compute_total_loss(target, synth, lambda_ps=1.0):
    # Loss of the PDF (pixels distribution)
    l_pdf = loss_pdf_cdf(target, synth)
    
    # Loss on the PS  (spatial structure)
    l_ps = loss_ps_log(target, synth)
    
    # Total loss
    total_loss = l_pdf + lambda_ps * l_ps
    return total_loss, l_pdf, l_ps

def get_target_baseline(target_folder):
    """Computes the mean and the standard deviation on the target patches"""
    paths = glob.glob(os.path.join(target_folder, "*.npy"))
    all_ps = []
    all_pdf = []
    
    for p in paths:
        img = np.load(p)
        # Power Spectrum in log for stability
        _, ps = power_spectrum(img)
        all_ps.append(np.log10(ps + 1e-10))
        
        # PDF (histogram bins)
        pdf, _ = np.histogram(img.flatten(), bins=50, range=(-10, 100), density=True)
        all_pdf.append(pdf)
        
    return {
        "ps_mean": np.mean(all_ps, axis=0),
        "ps_std": np.std(all_ps, axis=0),
        "pdf_mean": np.mean(all_pdf, axis=0),
        "pdf_std": np.std(all_pdf, axis=0)
    }

def compute_compatibility_score(synth_img, baseline):
    """Computes if the synthesis is in the 68% of the targets"""
    # Compute the stats of the synthesis
    _, ps_s = power_spectrum(synth_img)
    log_ps_s = np.log10(ps_s + 1e-10)
    pdf_s, _ = np.histogram(synth_img.flatten(), bins=50, range=(-10, 100), density=True)
    
    # Compute the reduced khi squarred for the power spectrum (divided by the std of the target)
    diff_ps = (log_ps_s - baseline["ps_mean"])**2 / (baseline["ps_std"]**2 + 1e-6)
    score_ps = np.mean(diff_ps)
    
    # Compute reduced khi squarred for the PDF
    diff_pdf = (pdf_s - baseline["pdf_mean"])**2 / (baseline["pdf_std"]**2 + 1e-6)
    score_pdf = np.mean(diff_pdf)
    
    # global score = average of the two terms
    total_score = (score_ps + score_pdf) / 2
    return total_score, score_ps, score_pdf