import torch
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from octis.models.n_functions import NFunction, PowerNFunction, ExpNFunction, ExpSquaredNFunction, LinearNFunction, ExpQuadraticQuarterNFunction, ExpHalfLinearCorrectedNFunction

def transform(X):
    """
    X: (N, d)
    
    Map x_i to (cos(h(x_i), sin(h(x_i)) * x_i))
    """
    return torch.cat([torch.cos(h(X)), torch.sin(h(X)) * X], dim=-1)

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


def osb_generate_trees_frames(ntrees, nlines, d, mean=128, std=0.1, device='cuda', gen_mode='gaussian_raw'):    
    # random root as gaussian distribution with given mean and std
    assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal'], "Invalid gen_mode"
    root = torch.randn(ntrees, 1, d, device=device) * std + mean
    intercept = root
    if gen_mode == 'gaussian_raw':
        theta = torch.randn(ntrees, nlines, d, device=device)
        theta = theta / torch.norm(theta, dim=-1, keepdim=True)
    elif gen_mode == 'gaussian_orthogonal':
        assert nlines <= d, "Support dim should be greater than or equal to number of lines to generate orthogonal lines"
        theta = torch.randn(ntrees, d, nlines, device=device)
        theta = svd_orthogonalize(theta)
        theta = theta.transpose(-2, -1)
    return theta, intercept


def svd_orthogonalize(matrix):
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U


class OSbSTSD():
    def __init__(self, ntrees=200, nlines=5, p=2, delta=2, device="cuda", type="normal", n_function="power", p_agg=2):
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
        return self.compute_via_taylor(h_edges, w_edges)
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
        else:
            raise ValueError("Unsupported N-function for Taylor GST")

        return (dist_per_tree.pow(self.p_agg).mean()).pow(1.0 / self.p_agg)
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

def sbsts(X, Y, ntrees=250, nlines=4, p=2, delta=2, device='cuda', type='normal'):
    TW_obj = OSbSTSD(ntrees=ntrees, nlines=nlines, p=p, delta=delta, device=device, type=type)
    stswd = TW_obj(X, Y)
    return stswd

def sbsts_unif(X, ntrees=250, nlines=4, p=2, delta=2, device='cuda', type='normal'):
    Y_unif = unif_hypersphere(X.shape, device=X.device) 
    stswd_unif = sbsts(X, Y_unif, ntrees, nlines, p, delta, device, type)
    return stswd_unif



