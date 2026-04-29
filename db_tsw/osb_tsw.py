import numpy as np
import torch

from db_tsw.n_functions import ExpHalfLinearCorrectedNFunction, ExpNFunction, ExpQuadraticQuarterNFunction, ExpSquaredNFunction, LinearNFunction, NFunction, PowerNFunction
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
            taylor_dist = self.compute_via_taylor(h_edges, w_edges)

            if self.optimization_method == "newton":
                op_dist = self.compute_via_optimization(h_edges, w_edges)

                eps = 1e-12
                rel_err = torch.abs(taylor_dist - op_dist) / (torch.abs(op_dist) + eps)

                print(
                    f"Taylor: {taylor_dist.item():.6e}, "
                    f"Optimization: {op_dist.item():.6e}, "
                    f"RelErr: {rel_err.item():.6e}"
                )

            return taylor_dist
    
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

            # Exact objective at k*
            z = k * h_flat.abs()
            exact_phi = self.n_function(z)
            loss_t = (1.0 + torch.sum(w_flat * exact_phi)) / k
            distances_per_tree.append(loss_t)

            # -----------------------------
            # Taylor tail at k*
            # -----------------------------
            if getattr(self, "print_tail_at_kstar", False):
                retained_phi = None

                if isinstance(self.n_function, ExpNFunction):
                    # Phi(t)=exp(t)-t-1
                    # retained: t^2/2 + t^3/6
                    retained_phi = 0.5 * z**2 + (1.0 / 6.0) * z**3

                elif isinstance(self.n_function, ExpSquaredNFunction):
                    # Phi(t)=exp(t^2)-1
                    # retained: t^2 + t^4/2
                    retained_phi = z**2 + 0.5 * z**4

                elif isinstance(self.n_function, ExpQuadraticQuarterNFunction):
                    # If Phi(t)=exp(t^2/4)-1
                    # retained: t^2/4 + t^4/32
                    retained_phi = 0.25 * z**2 + (1.0 / 32.0) * z**4

                elif isinstance(self.n_function, ExpHalfLinearCorrectedNFunction):
                    # If Phi(t)=0.5*(exp(t)-t-1)
                    # retained: t^2/4 + t^3/12
                    retained_phi = 0.25 * z**2 + (1.0 / 12.0) * z**3

                if retained_phi is not None:
                    tail_value = torch.sum(w_flat * (exact_phi - retained_phi)) / k
                    retained_obj = (1.0 + torch.sum(w_flat * retained_phi)) / k

                    tail_rel_exact = tail_value.abs() / (loss_t.abs() + 1e-12)
                    tail_rel_retained = tail_value.abs() / (retained_obj.abs() + 1e-12)

                    tail_rel_exact_list.append(tail_rel_exact.detach())
                    tail_rel_retained_list.append(tail_rel_retained.detach())
                    max_z_list.append(z.max().detach())
                    kstar_list.append(k.detach())

        dist_per_tree = torch.stack(distances_per_tree)
        out = (dist_per_tree.pow(self.p_agg).mean()).pow(1.0 / self.p_agg)

        # Print diagnostics aggregated over trees
        if getattr(self, "print_tail_at_kstar", False) and len(tail_rel_exact_list) > 0:
            tail_rel_exact_all = torch.stack(tail_rel_exact_list)
            tail_rel_retained_all = torch.stack(tail_rel_retained_list)
            max_z_all = torch.stack(max_z_list)
            kstar_all = torch.stack(kstar_list)

            print(
                "[Tail@k*] "
                f"tail/exact: mean={tail_rel_exact_all.mean().item():.6e}, "
                f"max={tail_rel_exact_all.max().item():.6e} | "
                f"tail/retained: mean={tail_rel_retained_all.mean().item():.6e}, "
                f"max={tail_rel_retained_all.max().item():.6e} | "
                f"max(k*h): mean={max_z_all.mean().item():.6e}, "
                f"max={max_z_all.max().item():.6e} | "
                f"k*: mean={kstar_all.mean().item():.6e}, "
                f"max={kstar_all.max().item():.6e}"
            )

        return out.to(dtype=orig_dtype, device=device)
    def orlicz_norm(self, d, max_iter=25, tol=1e-6):
        """
        Compute the Luxemburg Orlicz norm of a nonnegative vector d:
            ||d||_{L^Phi} = inf {lambda > 0 : mean Phi(d / lambda) <= 1}

        Uses binary search directly.
        Assumes:
        - self.n_function(x) computes Phi(x)
        - Phi is an N-function / Young function
        """
        eps = 1e-12
        d = torch.clamp(d, min=0.0)

        # Zero vector
        if torch.all(d <= eps):
            return torch.zeros((), device=d.device, dtype=d.dtype)

        def G(lmbda):
            return self.n_function(d / lmbda).mean()

        # Find bracket [lo, hi] such that
        # G(lo) >= 1 and G(hi) <= 1
        lo = torch.tensor(eps, device=d.device, dtype=d.dtype)
        hi = torch.clamp(d.max(), min=torch.tensor(1.0, device=d.device, dtype=d.dtype))

        # Expand hi until it is large enough
        g_hi = G(hi)
        expand_iter = 0
        while g_hi > 1.0 and expand_iter < 60:
            hi = hi * 2.0
            g_hi = G(hi)
            expand_iter += 1

        # Binary search
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            g_mid = G(mid)

            if g_mid > 1.0:
                lo = mid
            else:
                hi = mid

            # relative stopping
            if (hi - lo) / torch.clamp(hi, min=eps) < tol:
                break
        return hi

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

        else:
            raise ValueError("Unsupported N-function for Taylor GST")

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