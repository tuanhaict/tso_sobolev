import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
from methods.n_functions import EntropyLogNFunction, ExpHalfLinearCorrectedNFunction, ExpNFunction, ExpQuadraticQuarterNFunction, ExpSquaredNFunction, LinearNFunction, LogNFunction, OrliczNorm, PowerNFunction
from utils.func import transform

class OSbSTSD():
    def __init__(self, ntrees=200, nlines=5, p=2, delta=2, device="cuda", type="normal", n_function="power", p_agg=2, optimization_method="bounded"):
        """
        Class for computing the TW distance between two point clouds
        Args:
            mlp: The model that is used to divide the mass
            ntrees: Number of trees
            nlines: Number of lines per tree
            p: level of the norm
            delta: negative inverse of softmax temperature for distance based mass division
            device: device to run the code, follow torch convention
            n_function: type of N-function to use ("power", "exp", "exp_squared", "linear")
        """
        self.ntrees = ntrees
        self.nlines = nlines
        self.p = p
        self.delta = delta
        self.device = device
        self.eps = 1e-6
        self.p_agg = p_agg
        self.optimization_method = optimization_method
        if type not in ["normal", "generalized"]:
            raise ValueError("type should be either normal or generalized")
        self.type = type
        if n_function == "power":
            self.n_function = PowerNFunction(p)
        elif n_function == "exp":
            self.n_function = ExpNFunction()
        elif n_function == "exp_squared":
            self.n_function = ExpSquaredNFunction()
        elif n_function == "linear":
            self.n_function = LinearNFunction()
        elif n_function == "exp_squared_4":
            self.n_function = ExpQuadraticQuarterNFunction()
        elif n_function == "exp_2":
            self.n_function = ExpHalfLinearCorrectedNFunction()
        elif n_function == "log":
            self.n_function = LogNFunction()
        elif n_function == "entropy_log":
            self.n_function = EntropyLogNFunction()
        else:
            raise ValueError("Unsupported n_function type")
        self.use_closed_form = (isinstance(self.n_function, PowerNFunction) and 
                                 self.n_function.coeff == ((p-1)**(p-1))/(p**p))

    def __call__(self, X, Y):
        if self.type == "generalized":
            X = transform(X)
            Y = transform(Y)
            
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        root, intercept = self.generate_spherical_trees_frames(d=dn)
        
        combined_axis_coordinate, mass_X, mass_Y = self.get_mass_and_coordinate(X, Y, root, intercept)
        h_edges, w_edges = self.compute_edge_mass_and_weights(mass_X, mass_Y, combined_axis_coordinate)
        if self.use_closed_form:
            return self.compute_closed_form(h_edges, w_edges)
        else:
            dist = self.compute_via_original_root(h_edges, w_edges)

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
                    "maxiter": 200,
                },
            )

            k_star = np.exp(res.x)
            dist_t = res.fun

            dist_per_tree.append(dist_t)

        dist_per_tree = np.asarray(dist_per_tree, dtype=np.float64)

        out = np.mean(dist_per_tree ** self.p_agg) ** (1.0 / self.p_agg)

        return torch.tensor(out, device=orig_device, dtype=h_edges.dtype)
    def orlicz_norm(self, d, max_iter=25, tol=1e-6):
        return OrliczNorm.apply(d, self.n_function, max_iter, tol)
    def compute_via_taylor(self, h_edges, w_edges):
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
                + A4 / (2.0 * (A2 ).pow(1.5))
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
            dist_per_tree = (
                2.0 * torch.sqrt(A2)
                - A3 / (2.0 * A2)
                + A4 / (3.0 * (A2).pow(1.5))
                - A5 / (4.0 * (A2)**2)
            )
        elif isinstance(self.n_function, EntropyLogNFunction):
            A2 = torch.sum(w * h**2, dim=1)
            A3 = torch.sum(w * torch.abs(h)**3, dim=1)
            dist_per_tree = (
                torch.sqrt(2.0 *A2)
                - A3 / (3.0 * A2)
            )
        else:
            raise ValueError("Unsupported N-function for Taylor GST")
        return (dist_per_tree.pow(self.p_agg).mean()).pow(1.0 / self.p_agg)
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
        num_trees = h_edges.shape[0]
        device = h_edges.device

        distances_per_tree = []

        for t in range(num_trees):
            h = h_edges[t]        # (L, E)
            w = w_edges[t]        # (L, E)

            # Collapse lines + edges into one dimension (sum_e)
            h_flat = h.reshape(-1)
            w_flat = w.reshape(-1)

            # -----------------------------
            # Solve k*
            # -----------------------------
            with torch.no_grad():
                # init k using inverse mean scale
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
        # Mean over trees
        dist_per_tree = torch.stack(distances_per_tree)

        return (dist_per_tree.pow(self.p_agg).mean()).pow(1.0 / self.p_agg)
    def compute_edge_mass_and_weights(self, mass_X, mass_Y, combined_axis_coordinate):
        """
        Args:
            mass_X: (num_trees, num_lines, 2 * num_points)
            mass_Y: (num_trees, num_lines, 2 * num_points)
            combined_axis_coordinate: (num_trees, 2 * num_points)
        """
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_X.shape[0], mass_X.shape[1]
        indices = indices.unsqueeze(1).repeat(1, num_lines, 1)

        # generate the cumulative sum of mass
        mass_X_sorted = torch.gather(mass_X, 2, indices)
        mass_Y_sorted = torch.gather(mass_Y, 2, indices)
        sub_mass = mass_X_sorted - mass_Y_sorted
        sub_mass_cumsum = torch.cumsum(sub_mass, dim=-1)
        sub_mass_target_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_cumsum #(ntrees, nlines, 2*npoints)
        # generate the cumulative sum of length
        sub_length_target_cumsum = coord_sorted[..., -1:] - coord_sorted
        sub_length_target_cumsum = sub_length_target_cumsum.unsqueeze(1)

        ### compute edge length
        edge_length = torch.diff(coord_sorted, prepend=torch.zeros((num_trees, 1), device=coord_sorted.device), dim=-1)
        edge_length = edge_length.unsqueeze(1) #(ntrees, 1, 2*npoints)

        # h(e) is the absolute mass difference on each edge
        h_edges = torch.abs(sub_mass_target_cumsum)
        
        # w_e is the edge length
        w_edges = edge_length.repeat(1, num_lines, 1)
        
        return h_edges, w_edges
    def stw_concurrent_lines(self, mass_X, mass_Y, combined_axis_coordinate):
        """
        Args:
            mass_X: (num_trees, num_lines, 2 * num_points)
            mass_Y: (num_trees, num_lines, 2 * num_points)
            combined_axis_coordinate: (num_trees, 2 * num_points)
        """
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_X.shape[0], mass_X.shape[1]
        indices = indices.unsqueeze(1).repeat(1, num_lines, 1)

        # generate the cumulative sum of mass
        mass_X_sorted = torch.gather(mass_X, 2, indices)
        mass_Y_sorted = torch.gather(mass_Y, 2, indices)
        sub_mass = mass_X_sorted - mass_Y_sorted
        sub_mass_cumsum = torch.cumsum(sub_mass, dim=-1)
        sub_mass_target_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_cumsum #(ntrees, nlines, 2*npoints)

        ### compute edge length
        edge_length = torch.diff(coord_sorted, prepend=torch.zeros((num_trees, 1), device=coord_sorted.device), dim=-1)
        edge_length = edge_length.unsqueeze(1) #(ntrees, 1, 2*npoints)

        # compute TW distance
        subtract_mass = (torch.abs(sub_mass_target_cumsum) ** self.p) * edge_length
        subtract_mass_sum = torch.sum(subtract_mass, dim=[-1,-2])
        tw = torch.mean(subtract_mass_sum) ** (1/self.p)

        return tw

    def get_mass_and_coordinate(self, X, Y, root, intercept):
        # for the last dimension
        # 0, 1, 2, ...., N -1 is of distribution 1
        # N, N + 1, ...., 2N -1 is of distribution 2
        N, dn = X.shape
        mass_X, axis_coordinate_X = self.project(X, root=root, intercept=intercept)
        mass_Y, axis_coordinate_Y = self.project(Y, root=root, intercept=intercept)
        mass_X = torch.cat((mass_X, torch.zeros((mass_X.shape[0], mass_X.shape[1], N), device=self.device)), dim=2)
        mass_Y = torch.cat((torch.zeros((mass_Y.shape[0], mass_Y.shape[1], N), device=self.device), mass_Y), dim=2)

        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=-1)

        return combined_axis_coordinate, mass_X, mass_Y

    def project(self, input, root, intercept):
        """
        Args:
            input: (N, d)
            root: (ntrees, 1, d)
            intercept: (ntrees, nlines, d)
        
        Returns:
            mass_input: (ntrees, nlines, N)
            axis_coordinate: (ntrees, N)
        """
        N = input.shape[0]
        ntrees, nlines, d = intercept.shape
        # project input on great circle.
        input_alpha = root @ input.T #(ntrees, 1, N)
        input_pc = input - input_alpha.transpose(1, 2) @ root #(ntrees, N, d)
        input_pc = F.normalize(input_pc, p=2, dim=-1)
         
        ## get axis_coordinate 
        # coord based on distance from root to projections
        root_input_cosine = (root @ input.T).squeeze(1) #(ntrees, N) coordinate in vector root.
        axis_coordinate = torch.acos(torch.clamp(root_input_cosine, -1 + self.eps, 1 - self.eps)) #(ntrees, N)
        
        ## divide mass
        dist_cosine = intercept @ input_pc.transpose(1, 2) #(ntrees, nlines, N)
        dist = torch.acos(torch.clamp(dist_cosine, -1 + self.eps, 1 - self.eps)) 
        scale = torch.sin(axis_coordinate).unsqueeze(1) # (ntrees, 1, N)
        dist = dist * scale
        weight = -self.delta*dist #(ntrees, nlines, N)
        mass_input = torch.softmax(weight, dim=-2)/N

        return mass_input, axis_coordinate

    def generate_spherical_trees_frames(self, d):
        root = torch.randn(self.ntrees, 1, d, device=self.device)
        root = root / torch.norm(root, dim=-1, keepdim=True)
        # root = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample((self.ntrees, 1)).to(self.device)
        # root = root / torch.norm(root, dim=-1, keepdim=True)
        # intercept = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample((self.ntrees, self.nlines)).to(self.device)
        intercept = torch.randn(self.ntrees, self.nlines, d, device=self.device)
        intercept_proj = intercept @ root.transpose(1, 2) #(ntrees, nlines, 1)
        intercept = intercept - intercept_proj @ root #(ntrees, nlines, d)
        intercept = F.normalize(intercept, p=2, dim=-1)

        return root, intercept

def unif_hypersphere(shape, device):
    samples = torch.randn(shape, device=device)
    samples = F.normalize(samples, p=2, dim=-1)
    return samples

def osbsts(X, Y, ntrees=250, nlines=4, p=2, delta=2, device='cuda', type='normal', n_function="power", p_agg=2, optimization_method="bounded"):
    TW_obj = OSbSTSD(ntrees=ntrees, nlines=nlines, p=p, delta=delta, device=device, type=type, n_function=n_function, p_agg=p_agg, optimization_method=optimization_method)
    stswd = TW_obj(X, Y)
    return stswd

def osbsts_unif(X, ntrees=250, nlines=4, p=2, delta=2, device='cuda', type='normal'):
    Y_unif = unif_hypersphere(X.shape, device=X.device) 
    stswd_unif = osbsts(X, Y_unif, ntrees, nlines, p, delta, device, type)
    return stswd_unif