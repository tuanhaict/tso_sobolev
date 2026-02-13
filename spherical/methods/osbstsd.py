import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from methods.n_functions import ExpNFunction, ExpSquaredNFunction, LinearNFunction, PowerNFunction
from utils.func import transform

class OSbSTSD():
    def __init__(self, ntrees=200, nlines=5, p=2, delta=2, device="cuda", type="normal", n_function="power"):
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
        self.p_agg = 2
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

def osbsts(X, Y, ntrees=250, nlines=4, p=2, delta=2, device='cuda', type='normal', n_function="power"):
    TW_obj = OSbSTSD(ntrees=ntrees, nlines=nlines, p=p, delta=delta, device=device, type=type, n_function=n_function)
    stswd = TW_obj(X, Y)
    return stswd

def osbsts_unif(X, ntrees=250, nlines=4, p=2, delta=2, device='cuda', type='normal'):
    Y_unif = unif_hypersphere(X.shape, device=X.device) 
    stswd_unif = osbsts(X, Y_unif, ntrees, nlines, p, delta, device, type)
    return stswd_unif