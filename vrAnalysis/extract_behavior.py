"""
Standalone extraction of behavioral variables from VRBehavior_trial.mat files.

This module extracts VR behavioral data without requiring timeline or imaging
data. It handles both standard (version 1) and CR hippocannula (version 2)
vrControl file formats.
"""

import numpy as np
import scipy.io as scio
import scipy.sparse as sp
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VRBehaviorData:
    """Container for extracted VR behavioral data.

    Attributes with shape (S,) are per-sample (one entry per behavioral
    time-step across all trials concatenated). Attributes with shape (T,)
    are per-trial.

    Parameters
    ----------
    timestamps : np.ndarray, shape (S,)
        VR clock timestamps for each sample (seconds).
    position : np.ndarray, shape (S,)
        Virtual-track position for each sample.
    trial_index : np.ndarray, shape (S,)
        Trial number for each sample.
    trial_start_sample : np.ndarray, shape (T,)
        Index into the per-sample arrays where each trial begins.
    num_samples_per_trial : np.ndarray, shape (T,)
        Number of samples in each trial.
    room_length : np.ndarray, shape (T,)
        Track length for each trial.
    movement_gain : np.ndarray, shape (T,)
        Movement gain for each trial.
    reward_position : np.ndarray, shape (T,)
        Reward zone centre for each trial.
    reward_tolerance : np.ndarray, shape (T,)
        Reward zone half-width for each trial.
    reward_available : np.ndarray, shape (T,)
        Whether reward was available on each trial.
    reward_delivery_sample : np.ndarray, shape (T,)
        Sample index of reward delivery (-1 if no reward).
    active_licking : np.ndarray, shape (T,)
        Whether active licking was required on each trial.
    active_stopping : np.ndarray, shape (T,)
        Whether active stopping was required on each trial.
    environment_index : np.ndarray, shape (T,)
        Environment index for each trial (-1 if unavailable).
    lick_sample : np.ndarray, shape (L,)
        Sample indices at which licks occurred.
    num_trials : int
    num_samples : int
    mouse_name : str
    session_name : str
    date_str : str
    """

    # Per-sample arrays (S,)
    timestamps: np.ndarray
    position: np.ndarray
    trial_index: np.ndarray

    # Per-trial arrays (T,)
    trial_start_sample: np.ndarray
    num_samples_per_trial: np.ndarray
    room_length: np.ndarray
    movement_gain: np.ndarray
    reward_position: np.ndarray
    reward_tolerance: np.ndarray
    reward_available: np.ndarray
    reward_delivery_sample: np.ndarray
    active_licking: np.ndarray
    active_stopping: np.ndarray
    environment_index: np.ndarray

    # Lick events (L,)
    lick_sample: np.ndarray

    # Metadata
    num_trials: int
    num_samples: int
    mouse_name: str
    session_name: str
    date_str: str
    env_names: list  # list of str, indexed by environment_index; empty if unavailable
    reward_size_ul: float  # reward volume in µL (0 if unavailable)

    def get_trial(self, trial_idx):
        """Return (timestamps, position) arrays for a single trial."""
        start = self.trial_start_sample[trial_idx]
        end = start + self.num_samples_per_trial[trial_idx]
        return self.timestamps[start:end], self.position[start:end]

    def get_speed(self, trial_idx):
        """Return (position_midpoints, speed) for a single trial.

        Speed is computed as diff(position) / diff(time). The returned
        position array has length n-1 (midpoints between samples).
        """
        t, pos = self.get_trial(trial_idx)
        dt = np.diff(t)
        dp = np.diff(pos)
        speed = dp / dt
        pos_mid = (pos[:-1] + pos[1:]) / 2
        return pos_mid, speed

    def resample(self, rate):
        """Return a new VRBehaviorData resampled to a regular grid.

        Timestamps are zero-referenced per trial and interpolated to
        evenly spaced samples at the given rate. Position is linearly
        interpolated. Lick and reward-delivery sample indices are mapped
        to the nearest new sample.

        Parameters
        ----------
        rate : float
            Target sample rate in Hz (e.g. 10 for 10 Hz).

        Returns
        -------
        VRBehaviorData
        """
        dt = 1.0 / rate
        new_ts_list = []
        new_pos_list = []
        new_nsamp = []

        for ti in range(self.num_trials):
            t, pos = self.get_trial(ti)
            t0 = t - t[0]  # zero-reference
            t_new = np.arange(0, t0[-1], dt)
            pos_new = np.interp(t_new, t0, pos)
            new_ts_list.append(t_new)
            new_pos_list.append(pos_new)
            new_nsamp.append(len(t_new))

        new_nsamp = np.array(new_nsamp)
        new_start = np.concatenate([[0], np.cumsum(new_nsamp)[:-1]]).astype(np.int64)
        new_timestamps = np.concatenate(new_ts_list)
        new_position = np.concatenate(new_pos_list)
        new_trial_index = np.concatenate([i * np.ones(n, dtype=np.int64) for i, n in enumerate(new_nsamp)])

        # Remap reward delivery sample
        new_rd = np.empty(self.num_trials, dtype=np.int64)
        for ti in range(self.num_trials):
            rd = self.reward_delivery_sample[ti]
            if rd < 0:
                new_rd[ti] = -1
            else:
                # time of reward in zero-referenced trial time
                t_orig, _ = self.get_trial(ti)
                rd_local = rd - self.trial_start_sample[ti]
                rd_time = t_orig[rd_local] - t_orig[0]
                new_rd[ti] = new_start[ti] + int(np.argmin(np.abs(new_ts_list[ti] - rd_time)))

        # Remap lick samples
        new_lick = []
        for ls in self.lick_sample:
            ti = self.trial_index[ls]
            t_orig, _ = self.get_trial(ti)
            local_idx = ls - self.trial_start_sample[ti]
            lick_time = t_orig[local_idx] - t_orig[0]
            new_lick.append(new_start[ti] + int(np.argmin(np.abs(new_ts_list[ti] - lick_time))))
        new_lick = np.array(new_lick, dtype=np.int64)

        return VRBehaviorData(
            timestamps=new_timestamps,
            position=new_position,
            trial_index=new_trial_index,
            trial_start_sample=new_start,
            num_samples_per_trial=new_nsamp,
            room_length=self.room_length.copy(),
            movement_gain=self.movement_gain.copy(),
            reward_position=self.reward_position.copy(),
            reward_tolerance=self.reward_tolerance.copy(),
            reward_available=self.reward_available.copy(),
            reward_delivery_sample=new_rd,
            active_licking=self.active_licking.copy(),
            active_stopping=self.active_stopping.copy(),
            environment_index=self.environment_index.copy(),
            lick_sample=new_lick,
            num_trials=self.num_trials,
            num_samples=int(new_nsamp.sum()),
            mouse_name=self.mouse_name,
            session_name=self.session_name,
            date_str=self.date_str,
            env_names=self.env_names,
            reward_size_ul=self.reward_size_ul,
        )

    def save(self, save_dir):
        """Save all arrays as .npy files using the onedata naming convention.

        Parameters
        ----------
        save_dir : str or Path
            Directory to save into. Created if it doesn't exist.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "positionTracking.times": self.timestamps,
            "positionTracking.position": self.position,
            "positionTracking.trialIndex": self.trial_index,
            "trials.positionTracking": self.trial_start_sample,
            "trials.numSamples": self.num_samples_per_trial,
            "trials.environmentIndex": self.environment_index,
            "trials.roomlength": self.room_length,
            "trials.movementGain": self.movement_gain,
            "trials.rewardPosition": self.reward_position,
            "trials.rewardZoneHalfwidth": self.reward_tolerance,
            "trials.rewardAvailability": self.reward_available,
            "trials.rewardPositionTracking": self.reward_delivery_sample,
            "trials.activeLicking": self.active_licking,
            "trials.activeStopping": self.active_stopping,
            "licksTracking.positionTracking": self.lick_sample,
        }
        for name, arr in files.items():
            np.save(save_dir / f"{name}.npy", arr)

        print(f"Saved {len(files)} files to {save_dir}")


def _to_dense(data, num_trials):
    """Convert sparse or dense data to dense ndarray, truncated to num_trials rows."""
    data = data[:num_trials]
    if sp.issparse(data):
        return data.toarray().squeeze()
    return np.asarray(data).squeeze()


def extract_behavior(mat_path):
    """Extract behavioral variables from a VRBehavior_trial.mat file.

    Loads the .mat file, determines the vrControl version, and extracts all
    trial-level and sample-level behavioral variables. Timestamps are in the
    original VR clock (no timeline alignment).

    Parameters
    ----------
    mat_path : str or Path
        Path to a ``*_VRBehavior_trial.mat`` file.

    Returns
    -------
    VRBehaviorData
        Dataclass containing all extracted behavioral variables.
    """
    vr = scio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)

    # Detect version
    if "trialInfo" in vr:
        trialInfo = vr["trialInfo"]
        expInfo = vr["expInfo"]
        version = 1
    elif "TRIAL" in vr:
        trialInfo = vr["TRIAL"]
        expInfo = vr["EXP"]
        version = 2
    else:
        raise ValueError(f"Unrecognized mat file structure. Keys: {[k for k in vr if not k.startswith('__')]}")

    # Determine valid trials (rows with at least one nonzero/non-nan timestamp)
    time_raw = trialInfo.time
    if sp.issparse(time_raw):
        num_per_trial = np.diff(time_raw.tocsr().indptr)
    else:
        time_arr = np.asarray(time_raw)
        if np.any(np.isnan(time_arr)):
            num_per_trial = np.sum(~np.isnan(time_arr), axis=1)
        else:
            num_per_trial = np.sum(time_arr != 0, axis=1)

    valid = np.where(num_per_trial > 0)[0]
    num_trials = len(valid)
    if not np.array_equal(valid, np.arange(num_trials)):
        raise ValueError(f"Valid trials are not contiguous from 0: {valid}")

    def to_dense(data):
        return _to_dense(data, num_trials)

    # --- Per-sample data (numTrials x numSamples) -> list of 1-d arrays per trial ---
    time_dense = to_dense(trialInfo.time)
    pos_dense = to_dense(trialInfo.roomPosition)
    lick_dense = to_dense(trialInfo.lick)

    # Ensure 2-d even for single trial
    if time_dense.ndim == 1:
        time_dense = time_dense[np.newaxis, :]
        pos_dense = pos_dense[np.newaxis, :]
        lick_dense = lick_dense[np.newaxis, :]

    # Index of valid (non-zero / non-nan) samples per trial
    if np.any(np.isnan(time_dense)):
        nzindex = [np.where(~np.isnan(t))[0] for t in time_dense]
    else:
        nzindex = [np.nonzero(t)[0] for t in time_dense]

    timestamps_per_trial = [time_dense[i][nz] for i, nz in enumerate(nzindex)]
    position_per_trial = [pos_dense[i][nz] for i, nz in enumerate(nzindex)]
    lick_per_trial = [lick_dense[i][nz] for i, nz in enumerate(nzindex)]

    num_samples_per_trial = np.array([len(t) for t in timestamps_per_trial])
    timestamps = np.concatenate(timestamps_per_trial)
    position = np.concatenate(position_per_trial)

    trial_start_sample = np.concatenate([[0], np.cumsum(num_samples_per_trial)[:-1]]).astype(np.int64)
    trial_index = np.concatenate([i * np.ones(n, dtype=np.int64) for i, n in enumerate(num_samples_per_trial)])

    # --- Per-trial variables ---
    if version == 1:
        room_length = np.asarray(expInfo.roomLength[:num_trials], dtype=float)
        movement_gain = np.asarray(expInfo.mvmtGain[:num_trials], dtype=float)
        reward_position = to_dense(trialInfo.rewardPosition).astype(float)
        reward_tolerance = to_dense(trialInfo.rewardTolerance).astype(float)
        reward_available = to_dense(trialInfo.rewardAvailable).astype(bool)
        reward_delivery_frame = to_dense(trialInfo.rewardDeliveryFrame).astype(np.int64) - 1
        active_licking = to_dense(trialInfo.activeLicking).astype(bool)
        active_stopping = to_dense(trialInfo.activeStopping).astype(bool)
        env_index = (
            to_dense(trialInfo.vrEnvIdx).astype(np.int16)
            if "vrEnvIdx" in trialInfo._fieldnames
            else -1 * np.ones(num_trials, dtype=np.int16)
        )
    else:  # version 2
        room_length = np.ones(num_trials) * expInfo.roomLength
        movement_gain = np.ones(num_trials)
        reward_position = to_dense(trialInfo.trialRewPos).astype(float)
        reward_tolerance = np.full(num_trials, float(expInfo.rewPosTolerance))
        reward_available = to_dense(trialInfo.trialRewAvailable).astype(bool)
        reward_delivery_frame = to_dense(trialInfo.trialRewDelivery)
        reward_delivery_frame[np.isnan(reward_delivery_frame)] = 0
        reward_delivery_frame = reward_delivery_frame.astype(np.int64) - 1
        active_licking = to_dense(trialInfo.trialActiveLicking).astype(bool)
        active_stopping = to_dense(trialInfo.trialActiveStopping).astype(bool)
        env_index = (
            to_dense(trialInfo.vrEnvIdx).astype(np.int16)
            if "vrEnvIdx" in trialInfo._fieldnames
            else -1 * np.ones(num_trials, dtype=np.int16)
        )

    # Convert within-trial reward frame to global sample index
    reward_delivery_sample = np.array(
        [(rd + int(np.sum(num_samples_per_trial[:ti])) if rd >= 0 else -1) for ti, rd in enumerate(reward_delivery_frame)],
        dtype=np.int64,
    )

    # --- Lick events ---
    lick_samples = []
    for ti, licks in enumerate(lick_per_trial):
        frames = np.nonzero(licks)[0]
        counts = licks[frames].astype(int)
        offset = int(np.sum(num_samples_per_trial[:ti]))
        for f, c in zip(frames, counts):
            lick_samples.extend([f + offset] * c)
    lick_sample = np.array(lick_samples, dtype=np.int64)

    # --- Metadata ---
    mouse_name = str(getattr(expInfo, "animalName", ""))
    session_name = str(getattr(expInfo, "sessionName", ""))
    date_str = str(getattr(expInfo, "dateStr", ""))
    try:
        raw_names = getattr(expInfo, "envNames", None)
        if raw_names is not None:
            env_names = [str(s) for s in np.asarray(raw_names).flat]
        else:
            env_names = []
    except Exception:
        env_names = []

    try:
        reward_size_ul = float(getattr(expInfo, "rewardSize", 0))
    except Exception:
        reward_size_ul = 0.0

    return VRBehaviorData(
        timestamps=timestamps,
        position=position,
        trial_index=trial_index,
        trial_start_sample=trial_start_sample,
        num_samples_per_trial=num_samples_per_trial,
        room_length=room_length,
        movement_gain=movement_gain,
        reward_position=reward_position,
        reward_tolerance=reward_tolerance,
        reward_available=reward_available,
        reward_delivery_sample=reward_delivery_sample,
        active_licking=active_licking,
        active_stopping=active_stopping,
        environment_index=env_index,
        lick_sample=lick_sample,
        num_trials=num_trials,
        num_samples=len(timestamps),
        mouse_name=mouse_name,
        session_name=session_name,
        date_str=date_str,
        env_names=env_names,
        reward_size_ul=reward_size_ul,
    )
