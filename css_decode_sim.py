import numpy as np
from tqdm import tqdm
import json
import time
import datetime
from ldpc import BpOsdDecoder, BpDecoder
from bposd.css import css_code
import scipy


class css_decode_sim:
    """
    A class for simulating BP+OSD decoding of CSS codes
    Logical success/failure is determined ONLY by whether the residual has a logical component
    (syndrome checks are NOT used).
    """

    def __init__(self, hx=None, hz=None, **input_dict):
        # default input values
        default_input = {
            "error_rate": None,
            "xyz_error_bias": [1, 1, 1],
            "target_runs": 100,
            "seed": 0,
            "bp_method": "minimum_sum",
            "ms_scaling_factor": 0.625,
            "max_iter": 0,
            "osd_method": "osd_cs",
            "osd_order": 2,
            "save_interval": 2,
            "output_file": None,
            "check_code": 1,
            "tqdm_disable": 0,
            "run_sim": 1,
            # channel update scheme between X and Z decoding
            "channel_update": "alternate",  # "x->z", "z->x", "alternate", "none"
            "outer_iters": 5,
            "hadamard_rotate": 0,
            "hadamard_rotate_sector1_length": 0,
            "error_bar_precision_cutoff": 1e-3,
            "decoder_type": "bp",  # "bp_osd" or "bp"
            "bp_success_criterion": "LER",  # "LER" (logical only) or "WER_bp" (3-stage)
        }

        # apply defaults for keys not passed to the class
        for key in input_dict.keys():
            self.__dict__[key] = input_dict[key]
        for key in default_input.keys():
            if key not in input_dict:
                self.__dict__[key] = default_input[key]

        # Validate decoder_type
        if self.decoder_type not in ["bp_osd", "bp"]:
            raise ValueError(
                f"decoder_type must be 'bp_osd' or 'bp', got {self.decoder_type}"
            )

        # Validate channel_update
        valid_updates = {"x->z", "z->x", "alternate", "none"}
        if self.channel_update not in valid_updates:
            raise ValueError(
                f"channel_update must be one of {valid_updates}, got {self.channel_update}"
            )

        # Validate bp_success_criterion
        if self.bp_success_criterion not in ["LER", "WER_bp"]:
            raise ValueError(
                f"bp_success_criterion must be 'LER' or 'WER_bp', got {self.bp_success_criterion}"
            )

        # output variables
        output_values = {
            "K": None,
            "N": None,
            "start_date": None,
            "runtime": 0.0,
            "runtime_readable": None,
            "run_count": 0,
            "bp_converge_count_x": 0,
            "bp_converge_count_z": 0,
            "bp_syndrome_satisfied_count": 0,
            "bp_success_count": 0,
            "bp_nonconverge_count": 0,
            "bp_syndrome_fail_after_converge_count": 0,
            "bp_logical_error_after_syndrome_pass_count": 0,
            "bp_logical_error_count": 0,
            "bp_logical_error_rate": 0.0,
            "bp_logical_error_rate_eb": 0.0,
            "bp_word_error_rate": 0.0,
            "bp_word_error_rate_eb": 0.0,
            "osd0_success_count": 0,
            "osd0_logical_error_rate": 0.0,
            "osd0_logical_error_rate_eb": 0.0,
            "osd0_word_error_rate": 0.0,
            "osd0_word_error_rate_eb": 0.0,
            "osdw_success_count": 0,
            "osdw_logical_error_rate": 0.0,
            "osdw_logical_error_rate_eb": 0.0,
            "osdw_word_error_rate": 0.0,
            "osdw_word_error_rate_eb": 0.0,
            "min_logical_weight": 1e9,
        }

        for key in output_values.keys():
            if key not in self.__dict__:
                self.__dict__[key] = output_values[key]

        # attributes we wish to save
        temp = []
        for key in self.__dict__.keys():
            if key not in ["channel_probs_x", "channel_probs_z", "channel_probs_y", "hx", "hz"]:
                temp.append(key)
        self.output_keys = temp

        # RNG setup
        if self.seed == 0 or self.run_count != 0:
            self.seed = np.random.randint(low=1, high=2**32 - 1)
        np.random.seed(self.seed)
        print(f"RNG Seed: {self.seed}")
        print(f"BP Success Criterion: {self.bp_success_criterion}")

        # hx/hz matrices
        self.hx = scipy.sparse.csr_matrix(hx).astype(np.uint8)
        self.hz = scipy.sparse.csr_matrix(hz).astype(np.uint8)
        self.N = self.hz.shape[1]
        if self.min_logical_weight == 1e9:
            self.min_logical_weight = self.N

        self.error_x = np.zeros(self.N, dtype=np.uint8)
        self.error_z = np.zeros(self.N, dtype=np.uint8)

        # Construct CSS code
        self._construct_code()

        # Setup error channel
        self._error_channel_setup()

        # Setup decoders
        self._decoder_setup()

        if self.run_sim:
            self.run_decode_sim()

    def _create_decoder(self, parity_check_matrix, channel_probs):
        if self.decoder_type == "bp_osd":
            return BpOsdDecoder(
                parity_check_matrix,
                channel_probs=channel_probs,
                max_iter=self.max_iter,
                bp_method=self.bp_method,
                ms_scaling_factor=self.ms_scaling_factor,
                osd_method=self.osd_method,
                schedule="serial",
                osd_order=self.osd_order,
            )
        elif self.decoder_type == "bp":
            return BpDecoder(
                parity_check_matrix,
                channel_probs=channel_probs,
                max_iter=self.max_iter,
                bp_method=self.bp_method,
                schedule="serial",
                ms_scaling_factor=self.ms_scaling_factor,
            )
        raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

    def _get_decoding(self, decoder, decoding_type="osdw"):
        if self.decoder_type == "bp_osd":
            if decoding_type == "osdw":
                return np.asarray(decoder.osdw_decoding, dtype=np.uint8) % 2
            if decoding_type == "osd0":
                return np.asarray(decoder.osd0_decoding, dtype=np.uint8) % 2
            if decoding_type == "bp":
                return np.asarray(decoder.bp_decoding, dtype=np.uint8) % 2
            raise ValueError(f"Unknown decoding_type: {decoding_type}")
        elif self.decoder_type == "bp":
            return np.asarray(decoder.decoding, dtype=np.uint8) % 2
        raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

    def _single_run(self):
        # generate error
        self.error_x, self.error_z = self._generate_error()

        # reset channels to original (before any Bayesian updates)
        self.bpd_x.update_channel_probs(self.channel_probs_x + self.channel_probs_y)
        self.bpd_z.update_channel_probs(self.channel_probs_z + self.channel_probs_y)

        for _ in range(int(self.outer_iters)):
            # X decoding pass
            synd_x = np.asarray((self.hz @ self.error_x) % 2, dtype=np.uint8).reshape(-1)
            self.bpd_x.decode(synd_x)

            if self.channel_update in ("x->z", "alternate"):
                self._channel_update("x->z")

            # Z decoding pass
            synd_z = np.asarray((self.hx @ self.error_z) % 2, dtype=np.uint8).reshape(-1)
            self.bpd_z.decode(synd_z)

            if self.channel_update in ("z->x", "alternate"):
                self._channel_update("z->x")

        self._encoded_error_rates()

    def _channel_update(self, update_direction):
        if update_direction == "x->z":
            decoder_probs = np.zeros(self.N, dtype=float)
            osdw_decoding_x = self._get_decoding(self.bpd_x, "osdw")

            for i in range(self.N):
                if osdw_decoding_x[i] == 1:
                    denom = (self.channel_probs_x[i] + self.channel_probs_y[i])
                    decoder_probs[i] = 0.0 if denom == 0 else (self.channel_probs_y[i] / denom)
                else:
                    denom = 1.0 - self.channel_probs_x[i] - self.channel_probs_y[i]
                    decoder_probs[i] = 0.0 if denom <= 0 else (self.channel_probs_z[i] / denom)

            decoder_probs = np.clip(decoder_probs, 0.0, 1.0)
            self.bpd_z.update_channel_probs(decoder_probs)

        elif update_direction == "z->x":
            decoder_probs = np.zeros(self.N, dtype=float)
            osdw_decoding_z = self._get_decoding(self.bpd_z, "osdw")

            for i in range(self.N):
                if osdw_decoding_z[i] == 1:
                    denom = (self.channel_probs_z[i] + self.channel_probs_y[i])
                    decoder_probs[i] = 0.0 if denom == 0 else (self.channel_probs_y[i] / denom)
                else:
                    denom = 1.0 - self.channel_probs_z[i] - self.channel_probs_y[i]
                    decoder_probs[i] = 0.0 if denom <= 0 else (self.channel_probs_x[i] / denom)

            decoder_probs = np.clip(decoder_probs, 0.0, 1.0)
            self.bpd_x.update_channel_probs(decoder_probs)

    def _encoded_error_rates(self):
        """
        Success criterion (ALL decoders):
          Success <=> residual has NO logical component
          (syndrome checks are NOT used)
        """

        def _has_logical_error(res_x, res_z):
            # logical-Z overlap with X residual OR logical-X overlap with Z residual
            return (self.lz @ res_x % 2).any() or (self.lx @ res_z % 2).any()

        # -------- 1) OSDW --------
        dx_w = self._get_decoding(self.bpd_x, "osdw")
        dz_w = self._get_decoding(self.bpd_z, "osdw")
        residual_x = (self.error_x + dx_w) % 2
        residual_z = (self.error_z + dz_w) % 2

        log_fail = _has_logical_error(residual_x, residual_z)
        if not log_fail:
            self.osdw_success_count += 1
        else:
            # min logical weight tracking
            if (self.lz @ residual_x % 2).any():
                self.min_logical_weight = min(self.min_logical_weight, int(np.sum(residual_x)))
            if (self.lx @ residual_z % 2).any():
                self.min_logical_weight = min(self.min_logical_weight, int(np.sum(residual_z)))

        self.osdw_logical_error_rate = 1 - self.osdw_success_count / self.run_count
        self.osdw_logical_error_rate_eb = np.sqrt(
            (1 - self.osdw_logical_error_rate) * self.osdw_logical_error_rate / self.run_count
        )
        self.osdw_word_error_rate = 1.0 - (1 - self.osdw_logical_error_rate) ** (1 / self.K)
        self.osdw_word_error_rate_eb = (
            self.osdw_logical_error_rate_eb
            * ((1 - self.osdw_logical_error_rate_eb) ** (1 / self.K - 1))
            / self.K
        )

        # -------- 2) OSD0 --------
        dx_0 = self._get_decoding(self.bpd_x, "osd0")
        dz_0 = self._get_decoding(self.bpd_z, "osd0")
        residual_x_0 = (self.error_x + dx_0) % 2
        residual_z_0 = (self.error_z + dz_0) % 2

        log_fail_0 = _has_logical_error(residual_x_0, residual_z_0)
        if not log_fail_0:
            self.osd0_success_count += 1
        else:
            if (self.lz @ residual_x_0 % 2).any():
                self.min_logical_weight = min(self.min_logical_weight, int(np.sum(residual_x_0)))
            if (self.lx @ residual_z_0 % 2).any():
                self.min_logical_weight = min(self.min_logical_weight, int(np.sum(residual_z_0)))

        self.osd0_logical_error_rate = 1 - self.osd0_success_count / self.run_count
        self.osd0_logical_error_rate_eb = np.sqrt(
            (1 - self.osd0_logical_error_rate) * self.osd0_logical_error_rate / self.run_count
        )
        self.osd0_word_error_rate = 1.0 - (1 - self.osd0_logical_error_rate) ** (1 / self.K)
        self.osd0_word_error_rate_eb = (
            self.osd0_logical_error_rate_eb
            * ((1 - self.osd0_logical_error_rate_eb) ** (1 / self.K - 1))
            / self.K
        )

        # -------- 3) BP - CRITERION DEPENDS ON bp_success_criterion --------
        
        # ALWAYS track convergence for diagnostics
        bp_x_converged = hasattr(self.bpd_x, "converge") and self.bpd_x.converge
        bp_z_converged = hasattr(self.bpd_z, "converge") and self.bpd_z.converge
        bp_fully_converged = bp_x_converged and bp_z_converged
        
        if bp_x_converged:
            self.bp_converge_count_x += 1
        if bp_z_converged:
            self.bp_converge_count_z += 1
        
        # ALWAYS get BP decoding
        dx_bp = self._get_decoding(self.bpd_x, "bp")
        dz_bp = self._get_decoding(self.bpd_z, "bp")
        residual_x_bp = (self.error_x + dx_bp) % 2
        residual_z_bp = (self.error_z + dz_bp) % 2
        
        # ALWAYS check syndrome for diagnostics
        syndrome_x = np.asarray((self.hz @ residual_x_bp) % 2, dtype=np.uint8)
        syndrome_z = np.asarray((self.hx @ residual_z_bp) % 2, dtype=np.uint8)
        syndrome_satisfied = not (syndrome_x.any() or syndrome_z.any())
        
        if syndrome_satisfied:
            self.bp_syndrome_satisfied_count += 1
        
        # ALWAYS check logical error
        has_logical_error = _has_logical_error(residual_x_bp, residual_z_bp)
        
        # Track overall logical error count (regardless of criterion)
        if has_logical_error:
            self.bp_logical_error_count += 1
        
        # Now determine success based on criterion
        if self.bp_success_criterion == "LER":
            # LER: Only care about logical error (ignore convergence and syndrome)
            log_fail_bp = has_logical_error
            
        elif self.bp_success_criterion == "WER_bp":
            # Full 3-stage check - track each failure type independently
            if not bp_fully_converged:
                self.bp_nonconverge_count += 1
            if bp_fully_converged and not syndrome_satisfied:
                self.bp_syndrome_fail_after_converge_count += 1
            if bp_fully_converged and syndrome_satisfied and has_logical_error:
                self.bp_logical_error_after_syndrome_pass_count += 1
            
            # Overall failure: fails if ANY stage fails
            log_fail_bp = (not bp_fully_converged) or (not syndrome_satisfied) or has_logical_error
        
        # Update success count based on criterion
        if not log_fail_bp:
            self.bp_success_count += 1

        self.bp_logical_error_rate = 1 - self.bp_success_count / self.run_count
        self.bp_logical_error_rate_eb = np.sqrt(
            (1 - self.bp_logical_error_rate) * self.bp_logical_error_rate / self.run_count
        )
        self.bp_word_error_rate = 1.0 - (1 - self.bp_logical_error_rate) ** (1 / self.K)
        self.bp_word_error_rate_eb = (
            self.bp_logical_error_rate_eb
            * ((1 - self.bp_logical_error_rate_eb) ** (1 / self.K - 1))
            / self.K
        )

    def _construct_code(self):
        print("Constructing CSS code from hx and hz matrices...")
        if isinstance(self.hx, (np.ndarray, scipy.sparse.spmatrix)) and isinstance(
            self.hz, (np.ndarray, scipy.sparse.spmatrix)
        ):
            qcode = css_code(self.hx, self.hz)
            self.lx = qcode.lx
            self.lz = qcode.lz
            self.K = qcode.K
            self.N = qcode.N
            print("Checking the CSS code is valid...")
            if self.check_code and not qcode.test():
                raise Exception(
                    "Error: invalid CSS code. Check the form of your hx and hz matrices!"
                )
        else:
            raise Exception("Invalid object type for the hx/hz matrices")
        return None

    def _error_channel_setup(self):
        xyz_error_bias = np.array(self.xyz_error_bias)
        if xyz_error_bias[0] == np.inf:
            self.px = self.error_rate
            self.py = 0
            self.pz = 0
        elif xyz_error_bias[1] == np.inf:
            self.px = 0
            self.py = self.error_rate
            self.pz = 0
        elif xyz_error_bias[2] == np.inf:
            self.px = 0
            self.py = 0
            self.pz = self.error_rate
        else:
            self.px, self.py, self.pz = (
                self.error_rate * xyz_error_bias / np.sum(xyz_error_bias)
            )

        if self.hadamard_rotate == 0:
            self.channel_probs_x = np.ones(self.N) * (self.px)
            self.channel_probs_z = np.ones(self.N) * (self.pz)
            self.channel_probs_y = np.ones(self.N) * (self.py)

        elif self.hadamard_rotate == 1:
            n1 = self.hadamard_rotate_sector1_length
            self.channel_probs_x = np.hstack(
                [np.ones(n1) * (self.px), np.ones(self.N - n1) * (self.pz)]
            )
            self.channel_probs_z = np.hstack(
                [np.ones(n1) * (self.pz), np.ones(self.N - n1) * (self.px)]
            )
            self.channel_probs_y = np.ones(self.N) * (self.py)
        else:
            raise ValueError(
                f"The hadamard rotate attribute should be set to 0 or 1. Not '{self.hadamard_rotate}"
            )

        self.channel_probs_x.setflags(write=False)
        self.channel_probs_y.setflags(write=False)
        self.channel_probs_z.setflags(write=False)

    def _decoder_setup(self):
        self.ms_scaling_factor = float(self.ms_scaling_factor)

        # decoder for Z errors
        self.bpd_z = self._create_decoder(
            self.hx,
            channel_probs=self.channel_probs_z + self.channel_probs_y,
        )

        # decoder for X errors
        self.bpd_x = self._create_decoder(
            self.hz,
            channel_probs=self.channel_probs_x + self.channel_probs_y,
        )

    def _generate_error(self):
        for i in range(self.N):
            rand = np.random.random()
            if rand < self.channel_probs_z[i]:
                self.error_z[i] = 1
                self.error_x[i] = 0
            elif self.channel_probs_z[i] <= rand < (self.channel_probs_z[i] + self.channel_probs_x[i]):
                self.error_z[i] = 0
                self.error_x[i] = 1
            elif (self.channel_probs_z[i] + self.channel_probs_x[i]) <= rand < (
                self.channel_probs_x[i] + self.channel_probs_y[i] + self.channel_probs_z[i]
            ):
                self.error_z[i] = 1
                self.error_x[i] = 1
            else:
                self.error_z[i] = 0
                self.error_x[i] = 0

        return self.error_x, self.error_z

    def run_decode_sim(self):
        self.start_date = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%A, %B %d, %Y %H:%M:%S"
        )

        pbar = tqdm(
            range(self.run_count + 1, self.target_runs + 1),
            disable=self.tqdm_disable,
            ncols=0,
        )

        start_time = time.time()
        save_time = start_time

        for self.run_count in pbar:
            self._single_run()

            if self.decoder_type == "bp":
                if self.bp_success_criterion == "LER":
                    pbar.set_description(
                        f"d_max: {self.min_logical_weight}; "
                        f"BP_LER: {self.bp_logical_error_rate*100:.3g}±{self.bp_logical_error_rate_eb*100:.2g}%; "
                        f"LogErr: {self.bp_logical_error_count}/{self.run_count}; "
                        f"Success: {self.bp_success_count}/{self.run_count}; "
                        f"Conv_X: {self.bp_converge_count_x}/{self.run_count}; "
                        f"Conv_Z: {self.bp_converge_count_z}/{self.run_count}; "
                        f"Synd_OK: {self.bp_syndrome_satisfied_count}/{self.run_count}"
                    )
                else:  # WER_bp
                    pbar.set_description(
                        f"d_max: {self.min_logical_weight}; "
                        f"BP: {self.bp_logical_error_rate*100:.3g}±{self.bp_logical_error_rate_eb*100:.2g}%; "
                        f"Success: {self.bp_success_count}/{self.run_count}; "
                        f"NonConv: {self.bp_nonconverge_count}/{self.run_count}; "
                        f"SyndFail: {self.bp_syndrome_fail_after_converge_count}/{self.run_count}; "
                        f"LogErr: {self.bp_logical_error_after_syndrome_pass_count}/{self.run_count}; "
                        f"TotalLogErr: {self.bp_logical_error_count}/{self.run_count}; "
                        f"Conv_X: {self.bp_converge_count_x}/{self.run_count}; "
                        f"Conv_Z: {self.bp_converge_count_z}/{self.run_count}; "
                        f"Synd_OK: {self.bp_syndrome_satisfied_count}/{self.run_count}"
                    )
            else:
                pbar.set_description(
                    f"d_max: {self.min_logical_weight}; BP: {self.bp_logical_error_rate*100:.3g}±{self.bp_logical_error_rate_eb*100:.2g}%; "
                    f"OSDW_WER: {self.osdw_word_error_rate*100:.3g}±{self.osdw_word_error_rate_eb*100:.2g}%; "
                    f"OSDW: {self.osdw_logical_error_rate*100:.3g}±{self.osdw_logical_error_rate_eb*100:.2g}%; "
                    f"OSD0: {self.osd0_logical_error_rate*100:.3g}±{self.osd0_logical_error_rate_eb*100:.2g}%;"
                )

            current_time = time.time()
            save_loop = current_time - save_time

            if int(save_loop) > self.save_interval or self.run_count == self.target_runs:
                save_time = time.time()
                self.runtime = save_loop + self.runtime

                self.runtime_readable = time.strftime(
                    "%H:%M:%S", time.gmtime(self.runtime)
                )

                if self.output_file is not None:
                    with open(self.output_file, "w+") as f:
                        print(self.output_dict(), file=f)
        
        # Print final summary in succinct format
        if self.decoder_type == "bp":
            print("\n" + "="*80)
            if self.bp_success_criterion == "LER":
                print(f"FINAL: d_max: {self.min_logical_weight}; "
                      f"BP_LER: {self.bp_logical_error_rate*100:.3g}±{self.bp_logical_error_rate_eb*100:.2g}%; "
                      f"LogErr: {self.bp_logical_error_count}/{self.run_count}; "
                      f"Success: {self.bp_success_count}/{self.run_count}; "
                      f"Conv_X: {self.bp_converge_count_x}/{self.run_count}; "
                      f"Conv_Z: {self.bp_converge_count_z}/{self.run_count}; "
                      f"Synd_OK: {self.bp_syndrome_satisfied_count}/{self.run_count}")
            else:  # WER_bp
                print(f"FINAL: d_max: {self.min_logical_weight}; "
                      f"BP: {self.bp_logical_error_rate*100:.3g}±{self.bp_logical_error_rate_eb*100:.2g}%; "
                      f"Success: {self.bp_success_count}/{self.run_count}; "
                      f"NonConv: {self.bp_nonconverge_count}/{self.run_count}; "
                      f"SyndFail: {self.bp_syndrome_fail_after_converge_count}/{self.run_count}; "
                      f"LogErr: {self.bp_logical_error_after_syndrome_pass_count}/{self.run_count}; "
                      f"TotalLogErr: {self.bp_logical_error_count}/{self.run_count}; "
                      f"Conv_X: {self.bp_converge_count_x}/{self.run_count}; "
                      f"Conv_Z: {self.bp_converge_count_z}/{self.run_count}; "
                      f"Synd_OK: {self.bp_syndrome_satisfied_count}/{self.run_count}")
            print("="*80)

        return json.dumps(self.output_dict(), sort_keys=True, indent=4)

    def output_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            if key in self.output_keys:
                output_dict[key] = value
        return json.dumps(output_dict, sort_keys=True, indent=4)