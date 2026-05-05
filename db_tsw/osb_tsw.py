import numpy as np
import torch

from db_tsw.n_functions import EntropyLogNFunction, ExpHalfLinearCorrectedNFunction, ExpNFunction, ExpQuadraticQuarterNFunction, ExpSquaredNFunction, LinearNFunction, LogNFunction, NFunction, PowerNFunction
from db_tsw.utils import generate_trees_frames
from scipy.optimize import minimize_scalar

class OSb_TSConcurrentLines:
    """
    Generalized Distance-Based Tree Sliced Wasserstein with Orlicz geometry.
    
    Extends DbTSW to Orlicz geometric structure following the GST paper:
    "Generalized Sobolev Transport for Probability Measures on a Graph"
    """
    
    def __init__(self, n_function='exp', p=2, delta=2, 
                 mass_division='distance_based', device="cuda",
                 optimization_method='bounded', p_agg=2, i_max=6):
        """
        Args:
            n_function: Type of N-function. Options:
                - 'power': Phi(t) = ((p-1)^(p-1)/p^p) * t^p (generalizes standard DbTSW)
                - 'exp': Phi(t) = exp(t) - t - 1
                - 'exp_squared': Phi(t) = exp(t^2) - 1
                - 'linear': Phi(t) = t (W1 limit case)
                - NFunction object: custom N-function
            p: power for 'power' N-function (ignored for others)
            delta: negative inverse of softmax temperature for distance based mass division
            mass_division: how to divide mass, 'uniform' or 'distance_based'
            device: device to run the code
            optimization_method: 'bounded' or 'minimize_scalar' for univariate optimization
        """
        self.device = device
        self.p = p
        self.delta = delta
        self.mass_division = mass_division
        self.optimization_method = optimization_method
        self.p_agg = p_agg
        self.print_tail_at_kstar = True  # Set to True to print tail diagnostics at k*
        self.i_max = i_max
        assert self.mass_division in ['uniform', 'distance_based'], \
            "Invalid mass division. Must be one of 'uniform', 'distance_based'"
        
        # Initialize N-function
        if isinstance(n_function, NFunction):
            self.n_function = n_function
        elif n_function == 'power':
            self.n_function = PowerNFunction(p)
        elif n_function == 'exp':
            self.n_function = ExpNFunction()
        elif n_function == 'exp_squared':
            self.n_function = ExpSquaredNFunction()
        elif n_function == 'exp_squared_4':
            self.n_function = ExpQuadraticQuarterNFunction()
        elif n_function == 'exp_2':
            self.n_function = ExpHalfLinearCorrectedNFunction()
        elif n_function == 'linear':
            self.n_function = LinearNFunction()
        elif n_function == 'log':
            self.n_function = LogNFunction()
        elif n_function == 'entropy_log':
            self.n_function = EntropyLogNFunction()
        else:
            raise ValueError(f"Unknown n_function: {n_function}")
        
        # For power N-function with the specific coefficient, we have closed form
        self.use_closed_form = (isinstance(self.n_function, PowerNFunction) and 
                                 self.n_function.coeff == ((p-1)**(p-1))/(p**p))
    
    def __call__(self, X, Y, theta, intercept):
        """
        Compute Generalized DbTSW distance between X and Y.
        
        Args:
            X: tensor of shape (N, d) - first distribution
            Y: tensor of shape (M, d) - second distribution  
            theta: projection directions of shape (num_trees, num_lines, d)
            intercept: intercept point of shape (num_trees, 1, d)
        
        Returns:
            Generalized DbTSW distance (scalar)
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass and coordinates
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        
        combined_axis_coordinate, mass_XY = self.get_mass_and_coordinate(X, Y, theta, intercept)
        
        # Compute generalized tree Wasserstein
        gtw = self.compute_generalized_tw(mass_XY, combined_axis_coordinate)
        
        return gtw
    
    def compute_generalized_tw(self, mass_XY, combined_axis_coordinate):
        """
        Compute Generalized Tree Wasserstein using Orlicz geometry.
        
        IMPORTANT: With multiple trees, each tree has its own optimization variable k.
        
        For each tree t:
            GST_t = inf_{k>0} (1/k) * [1 + sum_e w_e * Phi(k * |h(e)|)]
        
        Final distance = mean(GST_1, GST_2, ..., GST_num_trees)
        
        Args:
            mass_XY: (num_trees, num_lines, 2 * num_points)
            combined_axis_coordinate: (num_trees, num_lines, 2 * num_points)
        
        Returns:
            Generalized TW distance (mean over trees)
        """
        # Sort coordinates and compute h(e) and w_e
        h_edges, w_edges = self.compute_edge_mass_and_weights(mass_XY, combined_axis_coordinate)
        
        # Check for closed form solution (Proposition 4.4)
        if self.use_closed_form:
            return self.compute_closed_form(h_edges, w_edges)
        else:
            dist = self.compute_via_original_root(h_edges, w_edges, max_iter=self.i_max)

            # if self.optimization_method == "newton":
            #     op_dist = self.compute_via_scipy_optimization(h_edges, w_edges)

            #     eps = 1e-12
            #     rel_err = torch.abs(dist - op_dist) / (torch.abs(op_dist) + eps)

            #     print(
            #         f"Original Root: {dist.item():.6e}, "
            #         f"Optimization: {op_dist.item():.6e}, "
            #         f"RelErr: {rel_err.item():.6e}"
            #     )

            return dist
    
    def compute_edge_mass_and_weights(self, mass_XY, combined_axis_coordinate):
        """
        Compute h(e) (mass difference on edges) and w_e (edge weights/lengths).
        
        This is adapted from the original tw_concurrent_lines method.
        
        Returns:
            h_edges: absolute mass differences |h(e)| of shape (num_trees, num_lines, num_edges)
            w_edges: edge weights/lengths of shape (num_trees, num_lines, num_edges)
        """
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_XY.shape[0], mass_XY.shape[1]
        
        # Generate cumulative sum of mass (this gives h at each point)
        sub_mass = torch.gather(mass_XY, 2, indices)
        sub_mass_target_cumsum = torch.cumsum(sub_mass, dim=-1)
        sub_mass_right_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_target_cumsum
        mask_right = torch.nonzero(coord_sorted > 0, as_tuple=True)
        sub_mass_target_cumsum[mask_right] = sub_mass_right_cumsum[mask_right]
        
        # Compute edge lengths
        root = torch.zeros(num_trees, num_lines, 1, device=self.device)
        root_indices = torch.searchsorted(coord_sorted, root)
        coord_sorted_with_root = torch.zeros(num_trees, num_lines, mass_XY.shape[2] + 1, device=self.device)
        edge_mask = torch.ones_like(coord_sorted_with_root, dtype=torch.bool)
        edge_mask.scatter_(2, root_indices, False)
        coord_sorted_with_root[edge_mask] = coord_sorted.flatten()
        edge_length = coord_sorted_with_root[:, :, 1:] - coord_sorted_with_root[:, :, :-1]
        
        # h(e) is the absolute mass difference on each edge
        h_edges = torch.abs(sub_mass_target_cumsum)
        
        # w_e is the edge length
        w_edges = edge_length
        
        return h_edges, w_edges
    
    def compute_closed_form(self, h_edges, w_edges):
        """
        Compute using closed form for Phi(t) = ((p-1)^(p-1)/p^p) * t^p.
        
        Following Proposition 4.4:
        For each tree: GST_tree = [sum_e w_e * |h(e)|^p]^(1/p)
        Final: mean over all trees
        """
        p = self.p
        # Sum over edges and lines for each tree separately
        weighted_sum_per_tree = torch.sum(w_edges * torch.pow(h_edges, p), dim=[-1, -2])  # (num_trees,)
        
        # Mean over trees
        distances_per_tree = torch.pow(weighted_sum_per_tree, 1 / p)

        return (distances_per_tree.pow(self.p_agg).mean()).pow(1 / self.p_agg)

    def compute_via_scipy_optimization(self, h_edges, w_edges, x_window=30.0):
        """
        Diagnostic-only optimization using scipy.optimize.minimize_scalar.

        This intentionally detaches from the PyTorch computation graph.
        Use only for printing / checking Taylor approximation, not as training loss.
        """

        import numpy as np
        from scipy.optimize import minimize_scalar

        orig_device = h_edges.device

        # detach completely: SciPy diagnostic should not enter graph
        h_all = h_edges.detach().double().cpu()
        w_all = w_edges.detach().double().cpu()

        num_trees = h_all.shape[0]
        dist_per_tree = []

        eps = 1e-12

        for t in range(num_trees):
            h = h_all[t].reshape(-1)
            w = w_all[t].reshape(-1)

            # remove zero-mass / zero-weight terms for numerical stability
            mask = (h > eps) & (w > eps)
            h = h[mask]
            w = w[mask]

            if h.numel() == 0:
                dist_per_tree.append(0.0)
                continue

            # good scale initialization
            k0 = 1.0 / (h.mean().item() + eps)
            x0 = np.log(k0)

            def objective_in_log_k(x):
                """
                Optimize over x = log k instead of k > 0 directly.
                This avoids positivity constraints and improves numerical stability.
                """
                k = np.exp(x)

                with torch.no_grad():
                    kh = k * h
                    Phi = self.n_function(kh)
                    val = (1.0 + torch.sum(w * Phi)) / k

                val = float(val.item())

                if not np.isfinite(val):
                    return np.inf

                return val

            res = minimize_scalar(
                objective_in_log_k,
                method="bounded",
                bounds=(x0 - x_window, x0 + x_window),
                options={
                    "xatol": 1e-10,
                    "maxiter": 10000,
                },
            )

            k_star = np.exp(res.x)
            dist_t = res.fun

            dist_per_tree.append(dist_t)

        dist_per_tree = np.asarray(dist_per_tree, dtype=np.float64)

        out = np.mean(dist_per_tree ** self.p_agg) ** (1.0 / self.p_agg)

        return torch.tensor(out, device=orig_device, dtype=h_edges.dtype)
    def compute_via_original_root(
        self,
        h_edges,
        w_edges,
        max_iter=6,
        k_min=1e-6,
        k_max=10000.0,
        bracket_factor=16.0,
        expand_steps=4,
        verbose=False,
    ):
        """
        Fast vectorized original-root solver.

        Solves:
            min_k (1 + sum_e w_e Phi(k h_e)) / k

        by solving:
            G(k) = sum_e w_e [z Phi'(z) - Phi(z)] - 1 = 0,
            z = k h_e.

        Uses local log-space bracket around leading scale k0, expands if needed,
        then fixed-iteration vectorized bisection.
        """

        orig_dtype = h_edges.dtype
        device = h_edges.device

        h = h_edges.reshape(h_edges.shape[0], -1)
        w = w_edges.reshape(w_edges.shape[0], -1)

        h_det = h.detach()
        w_det = w.detach()

        T = h.shape[0]
        eps = 1e-12

        A2 = torch.sum(w_det * h_det.square(), dim=1)
        valid = A2 > eps

        # Better leading scale depending on N-function.
        # These are only for bracketing, not final formula.
        if isinstance(self.n_function, ExpNFunction):
            k0 = torch.sqrt(2.0 / A2.clamp_min(eps))
        elif isinstance(self.n_function, EntropyLogNFunction):
            k0 = torch.sqrt(2.0 / A2.clamp_min(eps))
        else:
            k0 = 1.0 / torch.sqrt(A2.clamp_min(eps))

        k0 = torch.clamp(k0, min=k_min, max=k_max)

        global_lo = float(np.log(k_min))
        global_hi = float(np.log(k_max))
        log_factor = float(np.log(bracket_factor))

        lo_x = torch.clamp(torch.log(k0) - log_factor, min=global_lo, max=global_hi)
        hi_x = torch.clamp(torch.log(k0) + log_factor, min=global_lo, max=global_hi)
        def G_from_x(x):
            k = torch.exp(x)              # (T,)
            z = k[:, None] * h_det        # (T, LE)

            Phi = self.n_function(z)
            Phi_p = self.n_function.derivative(z)

            H = z * Phi_p - Phi
            H = torch.nan_to_num(H, nan=1e30, posinf=1e30, neginf=-1e30)

            G = torch.sum(w_det * H, dim=1) - 1.0
            G = torch.nan_to_num(G, nan=1e30, posinf=1e30, neginf=-1e30)

            return G

        with torch.no_grad():
            # --------------------------------------------------
            # Bracket root: need G(lo) <= 0 <= G(hi)
            # --------------------------------------------------
            G_lo = G_from_x(lo_x)
            G_hi = G_from_x(hi_x)

            for _ in range(expand_steps):
                need_left = valid & (G_lo > 0.0)
                need_right = valid & (G_hi < 0.0)

                # expand only trees whose bracket is bad
                lo_x = torch.where(
                    need_left,
                    torch.clamp(lo_x - log_factor, min=global_lo),
                    lo_x,
                )
                hi_x = torch.where(
                    need_right,
                    torch.clamp(hi_x + log_factor, max=global_hi),
                    hi_x,
                )

                G_lo = G_from_x(lo_x)
                G_hi = G_from_x(hi_x)

            bracketed = valid & (G_lo <= 0.0) & (G_hi >= 0.0)

            # Fallback to full global bracket for trees still not bracketed.
            bad = valid & (~bracketed)
            if bool(bad.any().item()):
                lo_x = torch.where(
                    bad,
                    torch.full_like(lo_x, global_lo),
                    lo_x,
                )
                hi_x = torch.where(
                    bad,
                    torch.full_like(hi_x, global_hi),
                    hi_x,
                )

                G_lo = G_from_x(lo_x)
                G_hi = G_from_x(hi_x)
                bracketed = valid & (G_lo <= 0.0) & (G_hi >= 0.0)

            if verbose:
                bad2 = int((valid & (~bracketed)).sum().item())
                if bad2 > 0:
                    print(
                        f"[WARNING] {bad2}/{T} trees still not bracketed. "
                        f"Will clip to endpoint. Increase k_max."
                    )

            # --------------------------------------------------
            # Fixed-iteration vectorized bisection.
            # No early break: avoids CPU-GPU sync.
            # --------------------------------------------------
            print(f"k_min={torch.exp(lo_x)}, k_max={torch.exp(hi_x)}")
            for _ in range(max_iter):
                mid_x = 0.5 * (lo_x + hi_x)
                G_mid = G_from_x(mid_x)

                go_right = G_mid < 0.0

                lo_x = torch.where(bracketed & go_right, mid_x, lo_x)
                hi_x = torch.where(bracketed & (~go_right), mid_x, hi_x)

            x_star = 0.5 * (lo_x + hi_x)

            # If not bracketed, choose endpoint based on sign.
            # If G_hi < 0, root is above hi -> use hi.
            # If G_lo > 0, root is below lo -> use lo.
            x_star = torch.where(
                bracketed & valid,
                x_star,
                torch.where(G_hi < 0.0, hi_x, lo_x),
            )

            k_star = torch.exp(x_star).detach()
            k_star = torch.where(valid, k_star, torch.ones_like(k_star))

        # --------------------------------------------------
        # Final objective with graph
        # --------------------------------------------------
        k_eval = k_star.to(dtype=h.dtype).clamp_min(k_min)

        z = k_eval[:, None] * h
        Phi = self.n_function(z)

        dist_per_tree = (1.0 + torch.sum(w * Phi, dim=1)) / k_eval

        valid_graph = torch.sum(w * h.square(), dim=1) > eps
        dist_per_tree = torch.where(
            valid_graph,
            dist_per_tree,
            torch.zeros_like(dist_per_tree),
        )

        out = (dist_per_tree.pow(self.p_agg).mean()).pow(1.0 / self.p_agg)

        if verbose and bool(valid.any().item()):
            print(
                f"[Original root robust] "
                f"k*: min={k_star[valid].min().item():.3e}, "
                f"max={k_star[valid].max().item():.3e}, "
                f"mean={k_star[valid].mean().item():.3e}"
            )

        return out.to(device=device, dtype=orig_dtype)


    def get_mass_and_coordinate(self, X, Y, theta, intercept):
        """
        Project X and Y onto trees/lines and compute masses and coordinates.
        """
        N, dn = X.shape
        mass_X, axis_coordinate_X = self.project(X, theta=theta, intercept=intercept)
        mass_Y, axis_coordinate_Y = self.project(Y, theta=theta, intercept=intercept)
        
        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=2)
        massXY = torch.cat((mass_X, -mass_Y), dim=2)
        
        return combined_axis_coordinate, massXY
    
    def project(self, input, theta, intercept):
        """
        Project points onto tree structure.
        """
        N, d = input.shape
        num_trees = theta.shape[0]
        num_lines = theta.shape[1]
        
        # Translate by intercept (root point)
        input_translated = (input - intercept)  # [T, B, D]
        
        # Project onto lines: axis_coordinate = theta · (input - intercept)
        axis_coordinate = torch.matmul(theta, input_translated.transpose(1, 2))
        input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate, theta)
        
        # Compute mass division
        if self.mass_division == 'uniform':
            mass_input = torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)
        elif self.mass_division == 'distance_based':
            dist = torch.norm(input_projected_translated - input_translated.unsqueeze(1), dim=-1)
            weight = -self.delta * dist
            mass_input = torch.softmax(weight, dim=-2) / N
        
        return mass_input, axis_coordinate
class OSbTSW(OSb_TSConcurrentLines):
    """Original DbTSW as special case of Generalized DbTSW with p-power N-function"""
    
    def __init__(self, p=2, delta=2, device="cuda", n_function='power'):
        super().__init__(
            p=p,
            delta=delta,
            device=device,
            mass_division='distance_based',
            n_function=n_function,
        )
if __name__ == "__main__":
    import argparse
    import torch
    from torch.profiler import profile, ProfilerActivity

    parser = argparse.ArgumentParser()

    parser.add_argument("--N", type=int, default=50000)
    parser.add_argument("--M", type=int, default=50000)
    parser.add_argument("--dn", type=int, default=1000)
    parser.add_argument("--dm", type=int, default=None)
    parser.add_argument("--ntrees", type=int, default=100)
    parser.add_argument("--nlines", type=int, default=10)
    parser.add_argument("--n_function", type=str, default='power')
    args = parser.parse_args()

    N = args.N
    M = args.M
    dn = args.dn
    dm = args.dm if args.dm is not None else args.dn
    ntrees = args.ntrees
    nlines = args.nlines
    device = args.device
    n_function = args.n_function

    TW_obj = torch.compile(OSbTSW(n_function=n_function)).to(device) if hasattr(OSbTSW(), "to") else torch.compile(OSbTSW(n_function=n_function))


    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)
    
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    torch.cuda.reset_peak_memory_stats(device=None)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        tw = TW_obj(X, Y, theta, intercept)

    prof.export_chrome_trace("trace_concurrent.json")
    with open("profile_result_concurrent.txt", "w") as f:
        table_str = prof.key_averages().table(sort_by="cpu_time_total", top_level_events_only=True)
        f.write(table_str)
        print(table_str)
    print(torch.cuda.max_memory_allocated(device=None) / 1024 / 1024)