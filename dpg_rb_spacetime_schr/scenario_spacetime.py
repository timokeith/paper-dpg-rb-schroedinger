"""
Problem setup: mesh, spaces, and operator assembly for the ultraweak space-time Schrödinger DPG system with parametrized potential
"""
import ngsolve as ng
from ngsolve import dx, x, y, exp, IfPos, sin, pi
from ngsolve.meshes import MakeStructured2DMesh

from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.parameters.functionals import ProjectionParameterFunctional

from complex_ngsolve_bindings import ComplexNGSolveMatrixOperator, ComplexNGSolveVectorSpace
from fom_spacetime import DPGModel
from utils_spacetime import d_x, Adj_A

QUAD_BONUS = 20  # additional quadrature order for the terms with the potentials and the right-hand side

# =============================================================================
# 1. MESH GENERATION AND FUNCTION SPACE CONSTRUCTION
# =============================================================================
def _create_mesh_and_spaces(p, dp, N):
    """
    Create NGSolve function spaces for Space-Time DPG formulation on a uniform N x N mesh.
    Domain: (0,1) x (0,1) where x is space, y is time.
    """
    print(f"  > Generating Structured Mesh ({N}x{N})...")
    mesh = MakeStructured2DMesh(nx=N, ny=N)

    # -------------------------------------------------------------------------
    # TRIAL SPACES
    # u in L2, q_plus (trace) in H1, q_prime (flux) in FacetFESpace
    # -------------------------------------------------------------------------
    U = ng.L2(mesh, order=p-1, complex=True)
    Q_plus = ng.H1(mesh, order=p, orderinner=0, complex=True, dirichlet="bottom|left|right")
    Q_prime = ng.FacetFESpace(mesh, order=p, complex=True)

    # Product Trial Space X = U x Q+ x Q'
    X = U * Q_plus * Q_prime

    # -------------------------------------------------------------------------
    # TEST SPACE
    # Y: discontinuous polynomials of the enriched degree p + dp
    # -------------------------------------------------------------------------
    Y = ng.L2(mesh, order=p+dp, complex=True)

    # Combined Space for Static Condensation (X x Y)
    XY = X * Y

    X_pm = ComplexNGSolveVectorSpace(X)
    Y_pm = ComplexNGSolveVectorSpace(Y)

    # Track horizontal Q_prime DOFs
    UDOFs, QplusDOFs = U.ndof, Q_plus.ndof
    qprime_offset = UDOFs + QplusDOFs

    horiz_qprime_dofs_XY = set()
    for edge in mesh.edges:
        verts = edge.vertices
        pts = [mesh[vertex].point for vertex in verts]
        if abs(pts[0][1] - pts[1][1]) < 1e-10: # horizontal edge
            dofs = Q_prime.GetDofNrs(ng.NodeId(ng.EDGE, edge.nr))
            for d in dofs:
                if d >= 0:
                    horiz_qprime_dofs_XY.add(qprime_offset + d)

    return mesh, X, Y, XY, X_pm, Y_pm, horiz_qprime_dofs_XY

# =============================================================================
# 2. PARAMETER DEFINITIONS
# =============================================================================
def _define_scenario_params(scenario):
    """
    Define parameter domains, potential components, and rhs function.
    """
    print(f"  > Setting up Case {scenario}")

    # =========================================================================
    # Case 1 (2 parameters)
    # =========================================================================
    if scenario == 1:
        ranges = {
            'mu1': (0.0, 150.0),
            'mu2': (0.0, 150.0)
        }

        V1 = exp(-200 * (x - 0.3)**2)
        V2 = exp(-200 * (x - 0.7)**2)

        pot_decomp = [('mu1', V1), ('mu2', V2)]

        rhs_expr = sin(pi * x) * sin(10.0 * y)

    # =========================================================================
    # Case 2 (3 parameters)
    # =========================================================================
    elif scenario == 2:
        ranges = {
            'mu1': (0.0, 200.0),
            'mu2': (0.0, 150.0),
            'mu3': (-40.0, 40.0)
        }

        V1 = (x - 0.5)**2
        V2 = exp(-100 * (x - 0.5)**2)
        V3 = (x - 0.5)

        pot_decomp = [('mu1', V1), ('mu2', V2), ('mu3', V3)]

        rhs_expr = exp(-150 * (x - 0.2 - 0.6 * y)**2) * sin(pi * y)

    else:
        raise ValueError(f"Unknown case {scenario}, only Cases 1 and 2 are defined")

    # NGSolve parameters of the FOM forms, set to the current parameter value in every FOM solve
    ng_params = {key: [ng.Parameter(0.5 * (r[0] + r[1]))] for key, r in ranges.items()}

    # Compute Mean Potential
    V_mean = 0
    for key, V_comp in pot_decomp:
        r = ranges.get(key)
        mu_val = 0.5 * (r[0] + r[1])
        V_mean += mu_val * V_comp

    return pot_decomp, V_mean, rhs_expr, ranges, ng_params

# =============================================================================
# 3. GRAMIAN ASSEMBLY (GRAPH NORM)
# =============================================================================
def _assemble_gramian_inverse(Y, Y_pm, V_mean):
    """
    Assemble the Gramian of Y in the graph norm of the mean-parameter potential and return its inverse as an operator.
    """
    e, v = Y.TrialFunction(), Y.TestFunction()

    print("  > Assembling Gramian using Mean Parameter Potential...")

    # Assemble (Using Mean Adj_A)
    g = ng.BilinearForm(Y, hermitian=True)
    g += e * ng.Conj(v) * dx
    g += Adj_A(e, V_mean) * ng.Conj(Adj_A(v, V_mean)) * dx(bonus_intorder=QUAD_BONUS)
    g.Assemble()

    inv_g = g.mat.Inverse(Y.FreeDofs(), inverse="pardiso")
    return ComplexNGSolveMatrixOperator(inv_g, Y_pm, Y_pm)

# =============================================================================
# 4. OPERATOR ASSEMBLY (AFFINE DECOMPOSITION) FOR ROM
# =============================================================================
def _assemble_B_operators(X, Y, X_pm, Y_pm, pot_decomp):
    """
    Assemble affine B operators.
    """
    u, qplus, qprime = X.TrialFunction()
    v = Y.TestFunction()

    n = ng.specialcf.normal(2)
    n_x, n_t = n[0], n[1]
    is_vert = IfPos(n_x**2 - n_t**2, 1.0, 0.0)
    is_horiz = IfPos(n_t**2 - n_x**2, 1.0, 0.0)

    ops_B = []
    coeffs_B = []

    # --- Parameter-Independent Part (B0) ---
    # Use Adj_A with V=0 for the derivative terms
    b0 = ng.BilinearForm(trialspace=X, testspace=Y)
    b0 += u * ng.Conj(Adj_A(v, V=0)) * dx

    # Boundary Terms
    b0 += (is_horiz * 1j * n_t * qplus * ng.Conj(v)) * dx(element_boundary=True)
    b0 += (is_vert * n_x * qplus * ng.Conj(d_x(v))) * dx(element_boundary=True)
    b0 += (is_vert * (-1.0) * n_x * qprime * ng.Conj(v)) * dx(element_boundary=True)
    b0.Assemble()

    ops_B.append(ComplexNGSolveMatrixOperator(b0.mat, Y_pm, X_pm))
    coeffs_B.append(1.0)

    # --- Affine Parts (Potential Terms) ---
    for key, V_comp in pot_decomp:
        bi = ng.BilinearForm(trialspace=X, testspace=Y)
        bi += u * ng.Conj(V_comp * v) * dx(bonus_intorder=QUAD_BONUS)
        bi.Assemble()

        ops_B.append(ComplexNGSolveMatrixOperator(bi.mat, Y_pm, X_pm))
        coeffs_B.append(ProjectionParameterFunctional(key, 1, 0))

    return LincombOperator(ops_B, coeffs_B)

def _assemble_rhs_operators(Y, Y_pm, rhs_expr):
    """
    Assemble the RHS functional f(v) = (source, v), which does not depend on the parameter in Cases 1 and 2.
    """
    ops_f = []
    coeffs_f = []
    v = Y.TestFunction()

    F = ng.LinearForm(Y)
    F += rhs_expr * ng.Conj(v) * dx(bonus_intorder=QUAD_BONUS)
    F.Assemble()

    gf_f = ng.GridFunction(Y)
    gf_f.vec.data = F.vec

    ops_f.append(VectorOperator(Y_pm.make_array([gf_f])))
    coeffs_f.append(1.0)

    return LincombOperator(ops_f, coeffs_f)

# =============================================================================
# 5. FOM BILINEAR/LINEAR FORM ASSEMBLY (FOR STATIC CONDENSATION)
# =============================================================================
def _assemble_fom_forms(XY, ng_params, rhs_expr, pot_decomp, V_mean):
    """
    Define the bilinear and the linear form of the mixed FOM system (they are assembled in DPGModel._compute).
    The Gramian block uses the mean-parameter reference inner product of Y (the same as op_G_inv for the ROM),
    so that the FOM is the practical DPG method with respect to the reference norm and its residual
    is a lower bound for the ROM residual. The B blocks carry the actual parameter-dependent potential.
    """
    u, qplus, qprime, e = XY.TrialFunction()
    w, qplus_test, qprime_test, v = XY.TestFunction()

    n = ng.specialcf.normal(2)
    n_x, n_t = n[0], n[1]
    is_vert = IfPos(n_x**2 - n_t**2, 1.0, 0.0)
    is_horiz = IfPos(n_t**2 - n_x**2, 1.0, 0.0)

    # Reconstruct V_total from ng_params
    V_total = 0
    for key, V_comp in pot_decomp:
        V_total += ng_params[key][0] * V_comp

    # --- Assembly ---
    a_full = ng.BilinearForm(XY, hermitian=True, condense=True)

    # 1. Block M (Gramian): (e, v) + (A*_mean e, A*_mean v) with the mean-parameter potential
    a_full += e * ng.Conj(v) * dx
    a_full += Adj_A(e, V_mean) * ng.Conj(Adj_A(v, V_mean)) * dx(bonus_intorder=QUAD_BONUS)

    # 2. Block B: (u, A* v)
    a_full += u * ng.Conj(Adj_A(v, V_total)) * dx(bonus_intorder=QUAD_BONUS)

    # Boundary terms for B
    term_B_1 = is_horiz * 1j * n_t * qplus * ng.Conj(v)
    term_B_2 = is_vert * n_x * qplus * ng.Conj(d_x(v))
    term_B_3 = is_vert * n_x * qprime * ng.Conj(v)
    a_full += (term_B_1 + term_B_2 - term_B_3) * dx(element_boundary=True)

    # 3. Block B^H: (A* e, w)
    a_full += Adj_A(e, V_total) * ng.Conj(w) * dx(bonus_intorder=QUAD_BONUS)

    term_BH_1 = is_horiz * (-1j) * n_t * ng.Conj(qplus_test) * e
    term_BH_2 = is_vert * n_x * ng.Conj(qplus_test) * d_x(e)
    term_BH_3 = is_vert  * n_x * ng.Conj(qprime_test) * e
    a_full += (term_BH_1 + term_BH_2 - term_BH_3) * dx(element_boundary=True)

    # RHS
    f_full = ng.LinearForm(XY)
    f_full += rhs_expr * ng.Conj(v) * dx(bonus_intorder=QUAD_BONUS)

    return a_full, f_full

# =============================================================================
# 6. PRODUCT FOR THE GRAM-SCHMIDT ORTHONORMALIZATION (scaled L2-like norm of X)
# =============================================================================
def _setup_L2_product(X, X_pm):
    """L2 product of u plus the element boundary products of q+ and q' (interior edges are counted twice).
    The factor 0.01, about the mesh size of the paper, balances the edge terms against the volume term."""
    u, qplus, qprime = X.TrialFunction()
    w, qplus_test, qprime_test = X.TestFunction()

    n = ng.specialcf.normal(2)
    is_vertical_edge = ng.IfPos(n[0]**2 - n[1]**2, 1.0, 0.0)

    norm_form = ng.BilinearForm(X, hermitian=True)
    norm_form += u * ng.Conj(w) * ng.dx
    norm_form += 0.01 * qplus * ng.Conj(qplus_test) * dx(element_boundary=True)
    norm_form += 0.01 * is_vertical_edge * qprime * ng.Conj(qprime_test) * dx(element_boundary=True)

    norm_form.Assemble()
    return ComplexNGSolveMatrixOperator(norm_form.mat, X_pm, X_pm)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def setup_scenario(p, dp, N, active_scenario = 1):
    """Full-order model of a case on a uniform N x N mesh, with its trial space, mesh and parameter ranges."""
    mesh, X, Y, XY, X_pm, Y_pm, horiz_dofs = _create_mesh_and_spaces(p, dp, N)

    pot_decomp, V_mean, rhs_expr, ranges, ng_params = _define_scenario_params(active_scenario)

    with ng.TaskManager():
        op_invG = _assemble_gramian_inverse(Y, Y_pm, V_mean)
        op_B = _assemble_B_operators(X, Y, X_pm, Y_pm, pot_decomp)
        op_f = _assemble_rhs_operators(Y, Y_pm, rhs_expr)

        a_full, f_full = _assemble_fom_forms(XY, ng_params, rhs_expr, pot_decomp, V_mean)

        L2_product =  _setup_L2_product(X, X_pm)

        fom = DPGModel(op_B, op_invG, op_f, X, XY, mesh, ng_params, a_full, f_full, V_mean,
                       horiz_qprime_dofs=horiz_dofs, products={'L2': L2_product})

    return fom, X, mesh, ranges
