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
                 optimization_method='bounded', p_agg=2):
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
            dist = self.compute_via_original_root(h_edges, w_edges)

            if self.optimization_method == "newton":
                op_dist = self.compute_via_scipy_optimization(h_edges, w_edges)

                eps = 1e-12
                rel_err = torch.abs(dist - op_dist) / (torch.abs(op_dist) + eps)

                print(
                    f"Original Root: {dist.item():.6e}, "
                    f"Optimization: {op_dist.item():.6e}, "
                    f"RelErr: {rel_err.item():.6e}"
                )

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
    
    def compute_via_optimization(self, h_edges, w_edges):
        """
        Compute using univariate optimization for general N-functions
        using implicit differentiation (detach k*).

        h_edges: (T, L, E)  |h(e)|, requires_grad=True
        w_edges: (T, L, E)  w_e, no grad
        """
        orig_dtype = h_edges.dtype
        device = h_edges.device

        h_edges = h_edges.double()
        w_edges = w_edges.double()

        num_trees = h_edges.shape[0]
        distances_per_tree = []

        # Diagnostics at k*
        tail_rel_exact_list = []
        tail_rel_retained_list = []
        max_z_list = []
        kstar_list = []

        for t in range(num_trees):
            h = h_edges[t]        # (L, E)
            w = w_edges[t]        # (L, E)

            h_flat = h.reshape(-1)
            w_flat = w.reshape(-1)

            # -----------------------------
            # Solve k*
            # -----------------------------
            with torch.no_grad():
                k = 1.0 / (h_flat.mean() + 1e-8)

                for _ in range(100):
                    kh = k * h_flat

                    Phi = self.n_function(kh)
                    Phi_p = self.n_function.derivative(kh)
                    Phi_pp = self.n_function.second_derivative(kh)

                    sum_Phi = torch.sum(w_flat * Phi)
                    sum_Phi_p = torch.sum(w_flat * h_flat * Phi_p)
                    sum_Phi_pp = torch.sum(w_flat * h_flat**2 * Phi_pp)

                    Fp = -(1.0 + sum_Phi) / k**2 + sum_Phi_p / k
                    Fpp = (
                        2.0 * (1.0 + sum_Phi) / k**3
                        - 2.0 * sum_Phi_p / k**2
                        + sum_Phi_pp / k
                    )

                    k = torch.clamp(k - Fp / (Fpp + 1e-12), min=1e-8)

            k = k.detach()
            kh = k * h_flat
            loss_t = (1.0 + torch.sum(w_flat * self.n_function(k * h_flat))) / k
            distances_per_tree.append(loss_t)

        dist_per_tree = torch.stack(distances_per_tree)
        out = (dist_per_tree.pow(self.p_agg).mean()).pow(1.0 / self.p_agg)

        return out.to(dtype=orig_dtype, device=device)
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

            print(
                f"[SciPy opt] tree={t:03d}, "
                f"k*={k_star:.6e}, "
                f"dist={dist_t:.6e}, "
                f"success={res.success}"
            )

        dist_per_tree = np.asarray(dist_per_tree, dtype=np.float64)

        out = np.mean(dist_per_tree ** self.p_agg) ** (1.0 / self.p_agg)

        return torch.tensor(out, device=orig_device, dtype=h_edges.dtype)
    def compute_via_original_root(
        self,
        h_edges,
        w_edges,
        max_iter=25,
        k_max=10000.0,
        k_tol=1e-4,
        g_tol=1e-8,
        verbose=False,
    ):
        """
        Faster torch-only root solver for the original univariate optimization:
            min_{k>0} (1 + sum_e w_e Phi(k h_e)) / k

        Solve first-order condition:
            G(k) = sum_e w_e [z_e Phi'(z_e) - Phi(z_e)] - 1 = 0,
        where z_e = k h_e.

        Assumes k* is not too large, e.g. k* <= k_max.
        k* is detached; final objective keeps graph via h_edges/w_edges.
        """

        orig_dtype = h_edges.dtype
        orig_device = h_edges.device

        h_all = h_edges.reshape(h_edges.shape[0], -1).double()
        w_all = w_edges.reshape(w_edges.shape[0], -1).double()

        eps = 1e-12
        num_trees = h_all.shape[0]
        dist_per_tree = []
        kstar_list = []

        # Candidate upper bounds. This avoids slow repeated doubling.
        candidate_bounds = []
        v = 1.0
        while v < k_max:
            candidate_bounds.append(v)
            v *= 2.0
        candidate_bounds.append(k_max)

        for t in range(num_trees):
            h_t = h_all[t]
            w_t = w_all[t]

            with torch.no_grad():
                h_det = h_t.detach()
                w_det = w_t.detach()

                mask = (h_det > eps) & (w_det > eps)
                h_s = h_det[mask]
                w_s = w_det[mask]

                if h_s.numel() == 0:
                    k_star = torch.tensor(
                        0.0,
                        device=orig_device,
                        dtype=torch.float64,
                    )
                    dist_per_tree.append(
                        torch.zeros((), device=orig_device, dtype=torch.float64)
                    )
                    kstar_list.append(k_star)
                    continue

                def G(k):
                    z = k * h_s
                    Phi = self.n_function(z)
                    Phi_p = self.n_function.derivative(z)
                    return torch.sum(w_s * (z * Phi_p - Phi)) - 1.0

                lo = torch.tensor(0.0, device=orig_device, dtype=torch.float64)

                # --------------------------------------------------
                # Fast bracket search using fixed candidate bounds
                # --------------------------------------------------
                hi = None
                g_hi = None

                for cand in candidate_bounds:
                    if cand > k_max:
                        cand = k_max

                    cand_t = torch.tensor(
                        cand,
                        device=orig_device,
                        dtype=torch.float64,
                    )

                    g_val = G(cand_t)

                    # If G(cand) is inf, it is certainly above zero.
                    if torch.isinf(g_val) and g_val > 0:
                        hi = cand_t
                        g_hi = g_val
                        break

                    # If finite and crosses zero, bracket found.
                    if torch.isfinite(g_val) and g_val >= 0.0:
                        hi = cand_t
                        g_hi = g_val
                        break

                # If still not bracketed, use k_max as fallback.
                if hi is None:
                    hi = torch.tensor(
                        k_max,
                        device=orig_device,
                        dtype=torch.float64,
                    )
                    g_hi = G(hi)

                    if verbose:
                        print(
                            f"[WARNING] tree={t}: G(k_max)={g_hi.item():.3e} < 0. "
                            f"k* may exceed k_max={k_max}."
                        )

                    # If k_max is not enough, use it as clipped k*.
                    # This should not happen if your observation k* <= k_max is true.
                    if torch.isfinite(g_hi) and g_hi < 0.0:
                        k_star = hi.detach()
                    else:
                        k_star = hi.detach()

                else:
                    # --------------------------------------------------
                    # Bisection with early stopping
                    # --------------------------------------------------
                    for _ in range(max_iter):
                        mid = 0.5 * (lo + hi)
                        g_mid = G(mid)

                        # Early stop by root residual
                        if torch.isfinite(g_mid) and torch.abs(g_mid) < g_tol:
                            lo = mid
                            hi = mid
                            break

                        if g_mid < 0.0:
                            lo = mid
                        else:
                            hi = mid

                        # Early stop by relative interval width
                        width = hi - lo
                        scale = torch.clamp(torch.abs(hi), min=1.0)

                        if width / scale < k_tol:
                            break

                    k_star = 0.5 * (lo + hi)

            k_star = k_star.detach()

            # --------------------------------------------------
            # Evaluate original objective with graph
            # --------------------------------------------------
            if k_star <= eps:
                loss_t = torch.zeros((), device=orig_device, dtype=torch.float64)
            else:
                z = k_star * h_t
                loss_t = (1.0 + torch.sum(w_t * self.n_function(z))) / k_star

            dist_per_tree.append(loss_t)
            kstar_list.append(k_star)

            if verbose:
                print(
                    f"[Original root fast] tree={t:03d}, "
                    f"k*={k_star.item():.6e}, "
                    f"dist={loss_t.detach().item():.6e}"
                )

        dist_per_tree = torch.stack(dist_per_tree)

        out = (dist_per_tree.pow(self.p_agg).mean()).pow(1.0 / self.p_agg)

        return out.to(device=orig_device, dtype=orig_dtype)

    def compute_via_taylor(self, h_edges, w_edges):
        eps = 1e-8

        # (T, L*E)
        h = h_edges.reshape(h_edges.shape[0], -1)
        w = w_edges.reshape(w_edges.shape[0], -1)

        if isinstance(self.n_function, PowerNFunction):
            p = self.p
            A_p = torch.sum(w * h**p, dim=1)

            Cp = (p - 1)**(1.0 / p) + (p - 1)**(-(p - 1) / p)
            dist_per_tree = Cp * (A_p).pow(1.0 / p)

        elif isinstance(self.n_function, ExpNFunction):
            A2 = torch.sum(w * h**2, dim=1)
            A3 = torch.sum(w * torch.abs(h)**3, dim=1)
            dist_per_tree = (
                torch.sqrt(2.0 * A2)
                + A3 / (3.0 * (A2))
            )

        elif isinstance(self.n_function, ExpSquaredNFunction):
            A2 = torch.sum(w * h**2, dim=1)
            A4 = torch.sum(w * h**4, dim=1)

            dist_per_tree = (
                2.0 * torch.sqrt(A2)
                + A4 / (2.0 * (A2).pow(1.5))
            )
        elif isinstance(self.n_function, LinearNFunction):
            dist_per_tree = torch.sum(w * torch.abs(h), dim=1)
        elif isinstance(self.n_function, ExpQuadraticQuarterNFunction):
            A2 = torch.sum(w * h**2, dim=1)
            A4 = torch.sum(w * h**4, dim=1)

            dist_per_tree = (
                torch.sqrt(A2)
                + A4 / (4.0 * (A2).pow(1.5))
            )

        elif isinstance(self.n_function, ExpHalfLinearCorrectedNFunction):
            A2 = torch.sum(w * h**2, dim=1)
            A3 = torch.sum(w * torch.abs(h)**3, dim=1)

            dist_per_tree = (
                torch.sqrt((A2)/2.0)
                + A3 / (6.0 * (A2))
            )
        elif isinstance(self.n_function, LogNFunction):
            A2 = torch.sum(w * h**2, dim=1)
            A3 = torch.sum(w * torch.abs(h)**3, dim=1)
            A4 = torch.sum(w * h**4, dim=1)
            A5 = torch.sum(w * torch.abs(h)**5, dim=1)
            A7 = torch.sum(w * torch.abs(h)**7, dim=1)
            A6 = torch.sum(w * h**6, dim=1)
            dist_per_tree = (
                2.0 * torch.sqrt(A2)
                - A3 / (2.0 * A2)
            )
        elif isinstance(self.n_function, EntropyLogNFunction):
            A2 = torch.sum(w * h**2, dim=1)
            A3 = torch.sum(w * torch.abs(h)**3, dim=1)
            A4 = torch.sum(w * h**4, dim=1)
            dist_per_tree = (
                torch.sqrt(2.0 *A2)
                - A3 / (3.0 * A2)
            )
        
        else:
            raise ValueError("Unsupported N-function for Taylor GST")
        if self.optimization_method == "newton":
            print(f"k* approx: {1/torch.sqrt(A2)}")
        return (dist_per_tree.pow(self.p_agg).mean()).pow(1.0 / self.p_agg)


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
class DbTSW(OSb_TSConcurrentLines):
    """Original DbTSW as special case of Generalized DbTSW with p-power N-function"""
    
    def __init__(self, p=2, delta=2, device="cuda"):
        super().__init__(
            n_function='power',
            p=p,
            delta=delta,
            device=device,
            mass_division='distance_based'
        )