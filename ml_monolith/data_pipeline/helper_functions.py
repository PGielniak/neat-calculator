# helper functions

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq


DT = 0.02  # 50 Hz
GRAVITY = 9.80665  # standard gravity in m/s²
FS = 50.0 # sampling frequency in Hz

def correlation(x, y):
    return np.corrcoef(x, y)[0,1] if np.std(x) > 0 and np.std(y) > 0 else 0

def energy(sig):
    """Calculate normalized energy of signal"""
    sig = np.asarray(sig)
    if len(sig) == 0:
        return 0.0
    return np.sum(sig**2) / len(sig)

def entropy(sig, bins=10):
    """Calculate entropy with safeguards against extreme values"""
    sig = np.asarray(sig)
    if len(sig) == 0 or np.std(sig) == 0:
        return 0.0
    
    hist, _ = np.histogram(sig, bins=bins, density=False)
    # Normalize to get probabilities
    hist = hist / hist.sum()
    # Filter out zeros and add small epsilon to prevent log(0)
    hist = hist[hist > 0]
    
    if len(hist) == 0:
        return 0.0
    
    # Clip entropy to reasonable range to prevent extreme values
    ent = -np.sum(hist * np.log2(hist + 1e-10))
    return np.clip(ent, 0, 10)  # entropy shouldn't exceed ~10 for 10 bins

def sma(x, y, z):
    x, y, z = map(np.asarray, (x, y, z))
    return np.sum(np.abs(x) + np.abs(y) + np.abs(z)) / len(x)

def mean_freq(spec, freqs):
    spec = np.asarray(spec)
    freqs = np.asarray(freqs)
    if spec.sum() == 0:
        return 0.0
    return np.sum(freqs * spec) / np.sum(spec)

def lowpass_filter(sig, cutoff=0.3, fs=FS, order=4):
    """Butterworth low-pass like UCI HAR (~0.3 Hz to isolate gravity)."""
    sig = np.asarray(sig)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, sig)

def angle_between(v1, v2):
    """Angle between 2 vectors in radians."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang)

def extract_features(window, dt=DT, fs=FS):
    """
    window: np.array (128, 6) [accX, accY, accZ, gyroX, gyroY, gyroZ]
    returns: dict of HAR-style features (Body + Gravity + Angles)
    """
    # *** NORMALIZE TO MATCH KAGGLE DATASET ***
    # Divide accelerometer by gravity to get values in 'g' units (like Kaggle: ~-1 to 1)
    accx = window[:, 0] / GRAVITY
    accy = window[:, 1] / GRAVITY
    accz = window[:, 2] / GRAVITY
    gyx = window[:, 3]
    gyy = window[:, 4]
    gyz = window[:, 5]
    
    features = {}

    # ---------- GRAVITY vs BODY SPLIT ----------
    gravity_x = lowpass_filter(accx, cutoff=0.3, fs=fs)
    gravity_y = lowpass_filter(accy, cutoff=0.3, fs=fs)
    gravity_z = lowpass_filter(accz, cutoff=0.3, fs=fs)

    body_x = accx - gravity_x
    body_y = accy - gravity_y
    body_z = accz - gravity_z

    # ---------- TIME-DOMAIN: BODY ACC & GYRO ----------
    time_signals = {
        "tBodyAcc-X": body_x,
        "tBodyAcc-Y": body_y,
        "tBodyAcc-Z": body_z,
        "tBodyGyro-X": gyx,
        "tBodyGyro-Y": gyy,
        "tBodyGyro-Z": gyz,
    }

    for name, sig in time_signals.items():
        sig = np.asarray(sig)
        base, axis = name.split('-')   # "tBodyAcc", "X"

        features[f"{base}-mean()-{axis}"]   = sig.mean()
        features[f"{base}-std()-{axis}"]    = sig.std()
        features[f"{base}-min()-{axis}"]    = sig.min()
        features[f"{base}-max()-{axis}"]    = sig.max()
        features[f"{base}-energy()-{axis}"] = energy(sig)
        features[f"{base}-entropy()-{axis}"] = entropy(sig)

    # ---------- TIME-DOMAIN: GRAVITY ACC ----------
    grav_signals = {
        "tGravityAcc-X": gravity_x,
        "tGravityAcc-Y": gravity_y,
        "tGravityAcc-Z": gravity_z,
    }

    for name, sig in grav_signals.items():
        sig = np.asarray(sig)
        base, axis = name.split('-')
        features[f"{base}-mean()-{axis}"]   = sig.mean()
        features[f"{base}-std()-{axis}"]    = sig.std()
        features[f"{base}-min()-{axis}"]    = sig.min()
        features[f"{base}-max()-{axis}"]    = sig.max()
        features[f"{base}-energy()-{axis}"] = energy(sig)
        features[f"{base}-entropy()-{axis}"] = entropy(sig)

    # ---------- TIME-DOMAIN: JERK (body acc + gyro) ----------
    body_x_j = np.diff(body_x) / dt
    body_y_j = np.diff(body_y) / dt
    body_z_j = np.diff(body_z) / dt

    gyx_j = np.diff(gyx) / dt
    gyy_j = np.diff(gyy) / dt
    gyz_j = np.diff(gyz) / dt

    jerk_time_signals = {
        "tBodyAccJerk-X": body_x_j,
        "tBodyAccJerk-Y": body_y_j,
        "tBodyAccJerk-Z": body_z_j,
        "tBodyGyroJerk-X": gyx_j,
        "tBodyGyroJerk-Y": gyy_j,
        "tBodyGyroJerk-Z": gyz_j,
    }

    for name, sig in jerk_time_signals.items():
        sig = np.asarray(sig)
        base, axis = name.split('-')
        features[f"{base}-mean()-{axis}"]   = sig.mean()
        features[f"{base}-std()-{axis}"]    = sig.std()
        features[f"{base}-min()-{axis}"]    = sig.min()
        features[f"{base}-max()-{axis}"]    = sig.max()
        features[f"{base}-energy()-{axis}"] = energy(sig)
        features[f"{base}-entropy()-{axis}"] = entropy(sig)

    # ---------- TIME-DOMAIN: MAGNITUDES ----------
    body_mag       = np.sqrt(body_x**2      + body_y**2      + body_z**2)
    grav_mag       = np.sqrt(gravity_x**2   + gravity_y**2   + gravity_z**2)
    body_jerk_mag  = np.sqrt(body_x_j**2    + body_y_j**2    + body_z_j**2)
    gyro_mag       = np.sqrt(gyx**2         + gyy**2         + gyz**2)
    gyro_jerk_mag  = np.sqrt(gyx_j**2       + gyy_j**2       + gyz_j**2)

    mag_signals = {
        "tBodyAccMag":      body_mag,
        "tGravityAccMag":   grav_mag,
        "tBodyAccJerkMag":  body_jerk_mag,
        "tBodyGyroMag":     gyro_mag,
        "tBodyGyroJerkMag": gyro_jerk_mag,
    }

    for base, sig in mag_signals.items():
        sig = np.asarray(sig)
        features[f"{base}-mean()"]   = sig.mean()
        features[f"{base}-std()"]    = sig.std()
        features[f"{base}-min()"]    = sig.min()
        features[f"{base}-max()"]    = sig.max()
        features[f"{base}-energy()"] = energy(sig)
        features[f"{base}-entropy()"] = entropy(sig)
        features[f"{base}-sma()"]    = np.sum(np.abs(sig)) / len(sig)

    # ---------- ANGLES (like UCI HAR) ----------
    body_acc_mean   = np.array([body_x.mean(),      body_y.mean(),      body_z.mean()])
    gravity_mean    = np.array([gravity_x.mean(),   gravity_y.mean(),   gravity_z.mean()])
    body_acc_j_mean = np.array([body_x_j.mean(),    body_y_j.mean(),    body_z_j.mean()])
    gyro_mean       = np.array([gyx.mean(),         gyy.mean(),         gyz.mean()])
    gyro_j_mean     = np.array([gyx_j.mean(),       gyy_j.mean(),       gyz_j.mean()])

    # angles between mean vectors and gravity
    features["angle(tBodyAccMean,gravityMean)"]      = angle_between(body_acc_mean, gravity_mean)
    features["angle(tBodyAccJerkMean,gravityMean)"]  = angle_between(body_acc_j_mean, gravity_mean)
    features["angle(tBodyGyroMean,gravityMean)"]     = angle_between(gyro_mean, gravity_mean)
    features["angle(tBodyGyroJerkMean,gravityMean)"] = angle_between(gyro_j_mean, gravity_mean)

    # angles between gravity and unit axes (approx HAR's angle(X,gravityMean), etc.)
    x_unit = np.array([1.0, 0.0, 0.0])
    y_unit = np.array([0.0, 1.0, 0.0])
    z_unit = np.array([0.0, 0.0, 1.0])

    features["angle(X,gravityMean)"] = angle_between(x_unit, gravity_mean)
    features["angle(Y,gravityMean)"] = angle_between(y_unit, gravity_mean)
    features["angle(Z,gravityMean)"] = angle_between(z_unit, gravity_mean)

    # ---------- FREQUENCY-DOMAIN (Body / Jerk / Mags as before) ----------
    N_acc  = len(body_x)
    N_j    = len(body_x_j)
    freqs_acc = rfftfreq(N_acc, d=dt)
    freqs_j   = rfftfreq(N_j, d=dt)

    freq_signals = {
        "fBodyAcc-X":      (np.abs(rfft(body_x)),      freqs_acc),
        "fBodyAcc-Y":      (np.abs(rfft(body_y)),      freqs_acc),
        "fBodyAcc-Z":      (np.abs(rfft(body_z)),      freqs_acc),

        "fBodyGyro-X":     (np.abs(rfft(gyx)),         freqs_acc),
        "fBodyGyro-Y":     (np.abs(rfft(gyy)),         freqs_acc),
        "fBodyGyro-Z":     (np.abs(rfft(gyz)),         freqs_acc),

        "fBodyAccJerk-X":  (np.abs(rfft(body_x_j)),    freqs_j),
        "fBodyAccJerk-Y":  (np.abs(rfft(body_y_j)),    freqs_j),
        "fBodyAccJerk-Z":  (np.abs(rfft(body_z_j)),    freqs_j),

        "fBodyGyroJerk-X": (np.abs(rfft(gyx_j)),       freqs_j),
        "fBodyGyroJerk-Y": (np.abs(rfft(gyy_j)),       freqs_j),
        "fBodyGyroJerk-Z": (np.abs(rfft(gyz_j)),       freqs_j),
    }

    for name, (spec, f) in freq_signals.items():
        base, axis = name.split('-')
        features[f"{base}-mean()-{axis}"]      = spec.mean()
        features[f"{base}-std()-{axis}"]       = spec.std()
        features[f"{base}-energy()-{axis}"]    = energy(spec)
        features[f"{base}-entropy()-{axis}"]   = entropy(spec)
        features[f"{base}-meanFreq()-{axis}"]  = mean_freq(spec, f)

    freq_mag_signals = {
        "fBodyAccMag":      (np.abs(rfft(body_mag)),       freqs_acc),
        "fBodyAccJerkMag":  (np.abs(rfft(body_jerk_mag)),  freqs_j),
        "fBodyGyroMag":     (np.abs(rfft(gyro_mag)),       freqs_acc),
        "fBodyGyroJerkMag": (np.abs(rfft(gyro_jerk_mag)),  freqs_j),
    }

    for base, (spec, f) in freq_mag_signals.items():
        features[f"{base}-mean()"]      = spec.mean()
        features[f"{base}-std()"]       = spec.std()
        features[f"{base}-energy()"]    = energy(spec)
        features[f"{base}-entropy()"]   = entropy(spec)
        features[f"{base}-meanFreq()"]  = mean_freq(spec, f)

    return features
