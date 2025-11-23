# gdsp_delay_comb
```python
"""
MODULE NAME:
gdsp_delay_comb.py

DESCRIPTION:
Fully differentiable fractional-delay feedback comb filter implemented in pure
functional JAX / GDSP style. The module exposes a variable-length delay line
with linear interpolation and smooth time-varying delay, feedback, and
feedforward parameters. All state is carried explicitly as tuples, making the
module compatible with gradient-based optimization and modern autodiff
workflows.

INPUTS:
- x : input sample (scalar JAX array) passed to delay_comb_tick()
- x_seq : input signal sequence (1D JAX array, shape [T]) passed to
          delay_comb_process()
- state : delay line state tuple created by delay_comb_init()
- params : parameter tuple
    (delay_samples_target,
     fbk_target,
     ffd_target,
     wet,
     dry,
     smooth_coeff)

OUTPUTS:
- y : output sample (scalar JAX array) from delay_comb_tick()
- y_seq : output signal sequence (1D JAX array) from delay_comb_process()
- new_state : updated state tuple

STATE VARIABLES:
(buffer, write_idx, delay_smooth, fbk_smooth, ffd_smooth, max_delay_samples)

where:
- buffer          : 1D JAX array of length N (delay buffer storing v[n])
- write_idx       : int32 scalar, write head index in [0, N-1]
- delay_smooth    : float32 scalar, smoothed delay length in samples
- fbk_smooth      : float32 scalar, smoothed feedback amount
- ffd_smooth      : float32 scalar, smoothed feedforward amount
- max_delay_samples : float32 scalar, maximum allowed delay in samples

EQUATIONS / MATH:

Comb core:
We implement a recirculating, fractional-delay comb with smoothed parameters.

1. Parameter smoothing (per sample):
   Let α = smooth_coeff in (0, 1).

   delay_target_clamped = clip(delay_samples_target, 1, max_delay_samples)
   fbk_target_clipped   = clip(fbk_target, -fbk_max, fbk_max)  with fbk_max < 1
   ffd_target_clipped   = ffd_target  (optionally clipped if desired)

   delay_smooth[n+1] = delay_smooth[n] + α * (delay_target_clamped - delay_smooth[n])
   fbk_smooth[n+1]   = fbk_smooth[n]   + α * (fbk_target_clipped   - fbk_smooth[n])
   ffd_smooth[n+1]   = ffd_smooth[n]   + α * (ffd_target_clipped   - ffd_smooth[n])

2. Fractional delay read (linear interpolation):

   We store the internal signal v[n] in a circular buffer with write index w[n].

   read_pos_raw = float(w[n]) - delay_smooth[n+1]
   read_pos     = mod(read_pos_raw, N)

   i0   = floor(read_pos)
   frac = read_pos - i0
   i1   = mod(i0 + 1, N)

   d0 = buffer[i0]
   d1 = buffer[i1]

   delayed_v = (1 - frac) * d0 + frac * d1

3. Comb feedback and feedforward:

   v[n] = x[n] + fbk_smooth[n+1] * delayed_v

   y[n] = dry * x[n] + wet * (delayed_v + ffd_smooth[n+1] * v[n])

4. State updates:

   buffer_next     = buffer with v[n] written at position w[n]
   w[n+1]          = mod(w[n] + 1, N)
   delay_smooth[n+1], fbk_smooth[n+1], ffd_smooth[n+1] as above

through-zero rules:
- Delay is expressed in samples and is clamped to the range
  [1, max_delay_samples]. Any attempt to push the delay to or below zero
  will smoothly stick at 1 sample. Negative or excessively large targets are
  smoothly projected into the allowed range via clipping, so the delay never
  crosses zero or the buffer length.

phase wrapping rules:
- The circular buffer write index and the fractional read position are
  wrapped via modular arithmetic:
    w[n+1]    = mod(w[n] + 1, N)
    read_pos  = mod(read_pos_raw, N)
    i1        = mod(i0 + 1, N)

nonlinearities:
- No explicit nonlinear waveshaping is applied; the system is linear
  aside from parameter clipping. All operations (add, mul, clip, mod) are
  differentiable almost everywhere, which is sufficient for gradient-based
  methods.

interpolation rules:
- Linear interpolation between adjacent delay-line samples implements a
  first-order Lagrange fractional-delay filter:
    delayed_v = (1 - frac)*buffer[i0] + frac*buffer[i1]
  This is everywhere differentiable in frac and in the buffer samples.

time-varying coefficient rules:
- delay_smooth, fbk_smooth, and ffd_smooth are updated at audio rate by
  a first-order smoothing filter, providing continuous-time, differentiable
  parameter trajectories. Time-variation of delay length and feeding
  coefficients is safe and click-free as long as smooth_coeff is chosen
  reasonably small.

NOTES:
- Stable parameter ranges:
    |fbk_target| < 1  ensures bounded feedback for constant delay.
    The implementation hard-clips fbk_target to [-0.999, 0.999].
- Delay range:
    1 <= delay_samples_target <= max_delay_samples
- All core operations are implemented using JAX primitives with no
  Python-side branching inside jitted functions and no dynamic allocations
  whose shapes depend on runtime values.
- State and params are tuples of JAX scalars/arrays; no dicts, classes,
  or dataclasses are used.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, jit


# ---------------------------------------------------------------------------
# 1. delay_comb_init(...)
# ---------------------------------------------------------------------------

def delay_comb_init(
    max_delay_seconds,
    init_delay_seconds,
    sample_rate,
    init_fbk=0.0,
    init_ffd=0.0,
):
    """
    Initialize the delay/comb filter state.

    Args:
        max_delay_seconds: maximum supported delay in seconds (Python float).
        init_delay_seconds: initial delay in seconds (Python float).
        sample_rate: sampling rate in Hz (Python float or int).
        init_fbk: initial feedback coefficient (Python float).
        init_ffd: initial feedforward coefficient (Python float).

    Returns:
        state: tuple
            (buffer, write_idx, delay_smooth, fbk_smooth, ffd_smooth,
             max_delay_samples)
    """
    # Compute sizes on Python side (outside jit)
    max_delay_samples = int(np.ceil(max_delay_seconds * float(sample_rate)))
    # Add a small margin for interpolation safety
    buffer_len = max_delay_samples + 2

    # Delay in samples (can be fractional)
    init_delay_samples = float(init_delay_seconds) * float(sample_rate)
    init_delay_samples = np.clip(init_delay_samples, 1.0, float(max_delay_samples))

    # Allocate buffer (static shape)
    buffer = jnp.zeros((buffer_len,), dtype=jnp.float32)

    write_idx = jnp.array(0, dtype=jnp.int32)
    delay_smooth = jnp.array(init_delay_samples, dtype=jnp.float32)
    fbk_smooth = jnp.array(init_fbk, dtype=jnp.float32)
    ffd_smooth = jnp.array(init_ffd, dtype=jnp.float32)
    max_delay_samples_jax = jnp.array(float(max_delay_samples), dtype=jnp.float32)

    state = (
        buffer,
        write_idx,
        delay_smooth,
        fbk_smooth,
        ffd_smooth,
        max_delay_samples_jax,
    )
    return state


# ---------------------------------------------------------------------------
# 2. delay_comb_update_state(...)
#     Smooth time-varying parameters (delay, feedback, feedforward)
# ---------------------------------------------------------------------------

@jit
def delay_comb_update_state(state, params):
    """
    Update smoothed parameters inside the state.

    Args:
        state: (buffer, write_idx, delay_smooth, fbk_smooth,
                ffd_smooth, max_delay_samples)
        params: (delay_samples_target, fbk_target, ffd_target,
                 wet, dry, smooth_coeff)

    Returns:
        new_state: updated state tuple with new smoothed parameters.
    """
    buffer, write_idx, delay_smooth, fbk_smooth, ffd_smooth, max_delay_samples = state
    delay_target, fbk_target, ffd_target, wet, dry, smooth_coeff = params

    # Hard stability bounds
    fbk_max = jnp.array(0.999, dtype=jnp.float32)

    # Clamp targets into valid ranges
    delay_target_clamped = jnp.clip(delay_target, 1.0, max_delay_samples)
    fbk_target_clipped = jnp.clip(fbk_target, -fbk_max, fbk_max)
    ffd_target_clipped = ffd_target  # can be further limited if desired

    alpha = smooth_coeff

    delay_smooth_new = delay_smooth + alpha * (delay_target_clamped - delay_smooth)
    fbk_smooth_new = fbk_smooth + alpha * (fbk_target_clipped - fbk_smooth)
    ffd_smooth_new = ffd_smooth + alpha * (ffd_target_clipped - ffd_smooth)

    new_state = (
        buffer,
        write_idx,
        delay_smooth_new,
        fbk_smooth_new,
        ffd_smooth_new,
        max_delay_samples,
    )
    return new_state


# ---------------------------------------------------------------------------
# 3. delay_comb_tick(x, state, params)
# ---------------------------------------------------------------------------

@jit
def delay_comb_tick(x, state, params):
    """
    Process a single input sample through the delay/comb filter.

    Args:
        x: scalar JAX array (input sample).
        state: current state tuple.
        params: parameter tuple
            (delay_samples_target, fbk_target, ffd_target,
             wet, dry, smooth_coeff)

    Returns:
        y: scalar JAX array (output sample).
        new_state: updated state tuple.
    """
    # First smooth parameters
    state_smoothed = delay_comb_update_state(state, params)
    (
        buffer,
        write_idx,
        delay_smooth,
        fbk_smooth,
        ffd_smooth,
        max_delay_samples,
    ) = state_smoothed

    delay_target, fbk_target, ffd_target, wet, dry, smooth_coeff = params

    # Buffer metadata (static shape)
    buffer_len = buffer.shape[0]
    buffer_len_f = jnp.array(float(buffer_len), dtype=jnp.float32)

    # Fractional delay read with linear interpolation
    write_idx_f = write_idx.astype(jnp.float32)
    read_pos_raw = write_idx_f - delay_smooth
    read_pos = jnp.mod(read_pos_raw, buffer_len_f)

    idx0_f = jnp.floor(read_pos)
    frac = read_pos - idx0_f
    idx0 = idx0_f.astype(jnp.int32)
    idx1 = jnp.mod(idx0 + 1, buffer_len)

    # Read two adjacent samples using dynamic_slice
    d0 = lax.dynamic_slice(buffer, (idx0,), (1,))
    d1 = lax.dynamic_slice(buffer, (idx1,), (1,))
    d0_val = d0[0]
    d1_val = d1[0]

    one = jnp.array(1.0, dtype=jnp.float32)
    delayed_v = (one - frac) * d0_val + frac * d1_val

    # Comb core
    v = x + fbk_smooth * delayed_v

    # Write updated v into buffer at current write index
    write_pos = jnp.mod(write_idx, buffer_len)
    v_vec = jnp.reshape(v, (1,))
    buffer_updated = lax.dynamic_update_slice(buffer, v_vec, (write_pos,))

    write_idx_new = jnp.mod(write_pos + 1, buffer_len).astype(jnp.int32)

    # Output mix
    y = dry * x + wet * (delayed_v + ffd_smooth * v)

    new_state = (
        buffer_updated,
        write_idx_new,
        delay_smooth,
        fbk_smooth,
        ffd_smooth,
        max_delay_samples,
    )
    return y, new_state


# ---------------------------------------------------------------------------
# 4. delay_comb_process(x, state, params)
# ---------------------------------------------------------------------------

@jit
def delay_comb_process(x, state, params):
    """
    Process an input sequence through the delay/comb filter using lax.scan.

    Args:
        x: 1D JAX array, shape [T], input signal.
        state: initial state tuple.
        params: parameter tuple
            (delay_samples_target, fbk_target, ffd_target,
             wet, dry, smooth_coeff)

    Returns:
        y: 1D JAX array, shape [T], output signal.
        final_state: state tuple after processing the sequence.
    """
    def body_fn(carry, x_t):
        y_t, new_carry = delay_comb_tick(x_t, carry, params)
        return new_carry, y_t

    final_state, y = lax.scan(body_fn, state, x)
    return y, final_state


# ---------------------------------------------------------------------------
# 5. Smoke test, plot example, listen example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
        have_sd = True
    except Exception:
        have_sd = False

    # Basic settings
    sample_rate = 48000.0
    duration = 1.0
    num_samples = int(sample_rate * duration)

    # Impulse input for visualization
    x_np = np.zeros((num_samples,), dtype=np.float32)
    x_np[0] = 1.0  # unit impulse
    x = jnp.asarray(x_np)

    # Initialize state
    max_delay_seconds = 0.1
    init_delay_seconds = 0.03
    init_fbk = 0.7
    init_ffd = 0.0

    state = delay_comb_init(
        max_delay_seconds=max_delay_seconds,
        init_delay_seconds=init_delay_seconds,
        sample_rate=sample_rate,
        init_fbk=init_fbk,
        init_ffd=init_ffd,
    )

    # Params (constant over time for this example)
    delay_samples_target = jnp.array(init_delay_seconds * sample_rate, dtype=jnp.float32)
    fbk_target = jnp.array(0.7, dtype=jnp.float32)
    ffd_target = jnp.array(0.0, dtype=jnp.float32)
    wet = jnp.array(1.0, dtype=jnp.float32)
    dry = jnp.array(0.0, dtype=jnp.float32)
    smooth_coeff = jnp.array(0.001, dtype=jnp.float32)

    params = (
        delay_samples_target,
        fbk_target,
        ffd_target,
        wet,
        dry,
        smooth_coeff,
    )

    # Process
    y, state_out = delay_comb_process(x, state, params)
    y_np = np.array(y)

    # Plot impulse response
    t_np = np.arange(num_samples) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(t_np, y_np, label="Impulse response")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("gdsp_delay_comb impulse response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Listen example (if sounddevice is available)
    if have_sd:
        print("Playing impulse response...")
        sd.play(y_np, int(sample_rate))
        sd.wait()
    else:
        print("sounddevice not available; skipping audio playback.")

# FOLLOWUP PROMPTS:
# - "Can you extend this module to support delay-time modulation with a JAX LFO input?"
# - "Can you refactor this delay_comb into a stereo version that shares parameters but keeps independent state per channel?"
# - "Show an example of using vmap with delay_comb_tick to process multiple parallel voices efficiently."

```
