# app.py

# ==============================================================================
# IMPORTS
# ==============================================================================
from flask import Flask, render_template, request, jsonify
import numpy as np
import scipy.optimize as spo
from scipy.optimize import minimize_scalar
import scipy.linalg
import traceback
from numpy.linalg import inv, norm, pinv

app = Flask(__name__)

# ==============================================================================
# CÁC HÀM GIẢI
# ==============================================================================

def solve_scipy_problem(user_code):
    """(1) Tối ưu tổng quát: Sử dụng trình giải SLSQP của Scipy, mạnh mẽ và đa năng."""
    safe_globals = {'np': np, 'spo': spo, '__builtins__': {'list': list, 'dict': dict, 'tuple': tuple, 'range': range, 'len': len, 'print': print}}
    local_env = {}; exec(user_code, safe_globals, local_env)
    if 'objective_func' not in local_env: raise ValueError("'objective_func' chưa được định nghĩa.")
    if 'x0' not in local_env: raise ValueError("'x0' chưa được định nghĩa.")
    objective_func, x0 = local_env['objective_func'], local_env['x0']
    constraints, bounds = local_env.get('constraints', ()), local_env.get('bounds', None)
    result = spo.minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return {
        'success': bool(result.success), 
        'message': result.message, 
        'solution': result.x.tolist(), # <--- ĐÃ SỬA LỖI: Đổi tên thành 'solution' cho nhất quán
        'objective_value': float(result.fun)
    }

def solve_with_simplex_tableau(c, A_eq, b_eq):
    """(2) QHTT - Đơn hình: Triển khai thuật toán Đơn hình 2 pha, trả về các bảng trung gian."""
    num_vars, num_constraints = len(c), len(A_eq)
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    tableau[:num_constraints, :num_vars] = A_eq; tableau[:num_constraints, num_vars:num_vars + num_constraints] = np.identity(num_constraints); tableau[:num_constraints, -1] = b_eq
    tableau[-1, num_vars:num_vars + num_constraints] = 1
    tableaus = []
    def format_tableau(tab, phase, iteration, message=""):
        header = ["Basis"] + [f"x{i+1}" for i in range(num_vars)] + [f"a{i+1}" for i in range(num_constraints)] + ["RHS"]
        html = f"<h4>Pha {phase} - Lần lặp {iteration} {message}</h4><table border='1' style='border-collapse: collapse; margin-bottom: 20px;'><thead><tr>" + "".join([f"<th>{h}</th>" for h in header]) + "</tr></thead><tbody>"
        basis_vars = []
        for i in range(num_constraints):
            row = tab[i, :-1]; ones = np.where(np.isclose(row, 1))[0]; others_are_zero = True
            if len(ones) == 1 and not np.isclose(np.sum(np.abs(tab[:-1, ones[0]])), 1): others_are_zero = False
            if len(ones) == 1 and others_are_zero: basis_vars.append(ones[0])
            else: basis_vars.append(-1)
        for i in range(num_constraints):
            basis_name = "Unknown";
            if basis_vars[i] != -1: basis_name = f"a{basis_vars[i] - num_vars + 1}" if basis_vars[i] >= num_vars else f"x{basis_vars[i] + 1}"
            html += f"<tr><td><b>{basis_name}</b></td>" + "".join([f"<td>{val:.2f}</td>" for val in tab[i, :]]) + "</tr>"
        html += "<tr><td><b>z</b></td>" + "".join([f"<td>{val:.2f}</td>" for val in tab[-1, :]]) + "</tr></tbody></table>"
        return html
    for i in range(num_constraints): tableau[-1, :] -= tableau[i, :]
    tableaus.append(format_tableau(tableau, 1, 1, "(Bảng khởi tạo Pha 1)"))
    for k in range(10):
        if not np.any(tableau[-1, :num_vars + num_constraints] < -1e-6): break
        pivot_col = np.argmin(tableau[-1, :num_vars + num_constraints])
        ratios = np.array([tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 1e-6 else np.inf for i in range(num_constraints)])
        if np.all(ratios == np.inf): return {"error": "Bài toán không giới nội."}
        pivot_row = np.argmin(ratios)
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col];
        for i in range(num_constraints + 1):
            if i != pivot_row: tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
        tableaus.append(format_tableau(tableau, 1, k + 2))
    if abs(tableau[-1, -1]) > 1e-6: return {"error": "Bài toán vô nghiệm.", "tableaus": tableaus}
    tableau = np.delete(tableau, slice(num_vars, num_vars + num_constraints), axis=1); tableau[-1, :] = 0; tableau[-1, :num_vars] = -np.array(c)
    for i in range(num_constraints):
        basis_col_index = np.where(np.isclose(tableau[i, :num_vars], 1))[0]
        if len(basis_col_index) == 1 and np.isclose(np.sum(np.abs(tableau[:-1, basis_col_index[0]])), 1):
            if not np.isclose(tableau[-1, basis_col_index[0]], 0): tableau[-1, :] -= tableau[-1, basis_col_index[0]] * tableau[i, :]
    def format_tableau_phase2(tab, iteration, message=""):
        header = ["Basis"] + [f"x{i+1}" for i in range(num_vars)] + ["RHS"]
        html = f"<h4>Pha 2 - Lần lặp {iteration} {message}</h4><table border='1' style='border-collapse: collapse; margin-bottom: 20px;'><thead><tr>" + "".join([f"<th>{h}</th>" for h in header]) + "</tr></thead><tbody>"
        basis_vars = [np.where(np.isclose(tab[i, :-1], 1))[0][0] if 1 in tab[i, :-1] else -1 for i in range(num_constraints)]
        for i in range(num_constraints):
            basis_name = "Unknown";
            if basis_vars[i] != -1: basis_name = f"x{basis_vars[i] + 1}"
            html += f"<tr><td><b>{basis_name}</b></td>" + "".join([f"<td>{val:.2f}</td>" for val in tab[i, :]]) + "</tr>"
        html += "<tr><td><b>z</b></td>" + "".join([f"<td>{val:.2f}</td>" for val in tab[-1, :]]) + "</tr></tbody></table>"
        return html
    tableaus.append(format_tableau_phase2(tableau, 1, "(Bảng khởi tạo Pha 2)"))
    for k in range(10):
        if not np.any(tableau[-1, :num_vars] < -1e-6): break
        pivot_col = np.argmin(tableau[-1, :num_vars])
        ratios = np.array([tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 1e-6 else np.inf for i in range(num_constraints)])
        if np.all(ratios == np.inf): return {"error": "Bài toán không giới nội."}
        pivot_row = np.argmin(ratios); tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(num_constraints + 1):
            if i != pivot_row: tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
        tableaus.append(format_tableau_phase2(tableau, k + 2))
    solution = np.zeros(num_vars)
    for i in range(num_constraints):
        basis_col_index = np.where(np.isclose(tableau[i, :num_vars], 1))[0]
        if len(basis_col_index) == 1 and np.isclose(np.sum(np.abs(tableau[:-1, basis_col_index[0]])), 1):
            solution[basis_col_index[0]] = tableau[i, -1]
    return {"success": True, "message": "Đã tìm thấy lời giải tối ưu.", "solution": solution.tolist(), "objective_value": -tableau[-1, -1], "tableaus": tableaus}

def solve_nlp_barrier_method(user_code):
    """(3) NLP - Điểm trong/Hàm chắn: Giải NLP có ràng buộc BĐT bằng Barrier Method."""
    safe_globals = {'np': np, 'norm': norm, '__builtins__': {'list': list, 'dict': dict, 'tuple': tuple, 'range': range, 'len': len, 'print': print}}
    local_env = {}; exec(user_code, safe_globals, local_env)
    f, grad_f = local_env.get('objective_func'), local_env.get('grad_func')
    g_funcs, g_grads = local_env.get('ineq_constraint_funcs'), local_env.get('ineq_jacobian_funcs')
    x_k, mu_k, beta = np.array(local_env.get('x0'), dtype=float), local_env.get('mu0', 1.0), local_env.get('beta', 0.1)
    if not all([f, grad_f, g_funcs, g_grads, x_k is not None]): raise ValueError("Cần định nghĩa đầy đủ các hàm và biến.")
    max_iter, tolerance, history = 30, 1e-6, []
    for k in range(max_iter):
        for i, g_i in enumerate(g_funcs):
            if g_i(x_k) >= 0: return {"success": False, "message": f"Lỗi ở bước lặp {k}: Điểm x_k vi phạm ràng buộc g_{i}(x) < 0.", "history": history}
        history.append({"iteration": k, "mu_k": mu_k, "x_k": x_k.tolist(), "f_x_k": f(x_k)})
        if mu_k < tolerance: return {"success": True, "message": f"Thuật toán hội tụ sau {k} bước.", "solution": x_k.tolist(), "objective_value": f(x_k), "history": history}
        def barrier_objective(x): return f(x) - mu_k * np.sum([np.log(-g_i(x)) for g_i in g_funcs])
        def barrier_gradient(x): return grad_f(x) - mu_k * np.sum([g_grad_i(x) / g_i(x) for g_i, g_grad_i in zip(g_funcs, g_grads)], axis=0)
        res = spo.minimize(barrier_objective, x_k, method='BFGS', jac=barrier_gradient, tol=1e-5)
        x_k, mu_k = res.x, mu_k * beta
    return {"success": False, "message": f"Không hội tụ sau {max_iter} bước.", "solution": x_k.tolist(), "objective_value": f(x_k), "history": history}

def solve_qp_null_space(G, d, A, b):
    """(4) QP - Không gian Null: Giải QP có ràng buộc đẳng thức bằng cách chiếu vào không gian null."""
    G = np.array(G, dtype=float); d = np.array(d, dtype=float); A = np.array(A, dtype=float); b = np.array(b, dtype=float)
    try: x_p = np.linalg.pinv(A) @ b
    except np.linalg.LinAlgError: return {"success": False, "message": "Lỗi: Không thể tìm giải pháp riêng."}
    Z = scipy.linalg.null_space(A)
    if Z.shape[1] == 0:
        obj_val = 0.5 * x_p.T @ G @ x_p + d.T @ x_p
        return {"success": True, "message": "Không gian null rỗng.", "solution": x_p.tolist(), "objective_value": obj_val, "xp": x_p.tolist(), "Z": "Rỗng.", "v_star": "N/A"}
    G_reduced = Z.T @ G @ Z; d_reduced = (d.T @ Z + x_p.T @ G @ Z).T
    try: v_star = np.linalg.solve(G_reduced, -d_reduced)
    except np.linalg.LinAlgError: return {"success": False, "message": "Ma trận Hessian rút gọn suy biến.", "xp": x_p.tolist(), "Z": Z.tolist()}
    x_star = x_p + Z @ v_star; obj_val = 0.5 * x_star.T @ G @ x_star + d.T @ x_star
    return {"success": True, "message": "Đã tìm thấy lời giải tối ưu.", "solution": x_star.tolist(), "objective_value": obj_val, "xp": x_p.tolist(), "Z": Z.tolist(), "v_star": v_star.tolist()}

def solve_qp_active_set(G, d, A_eq, b_eq, A_ineq, b_ineq, x0):
    """(5) QP - Tập hoạt động: Giải QP có cả ràng buộc đẳng thức và bất đẳng thức."""
    G=np.array(G, dtype=float); d=np.array(d, dtype=float); x_k = np.array(x0, dtype=float)
    has_eq = A_eq is not None and len(A_eq) > 0; has_ineq = A_ineq is not None and len(A_ineq) > 0
    if has_eq: A_eq = np.array(A_eq, dtype=float); b_eq = np.array(b_eq, dtype=float)
    if has_ineq: A_ineq = np.array(A_ineq, dtype=float); b_ineq = np.array(b_ineq, dtype=float)
    working_set = []
    if has_ineq:
        for i in range(A_ineq.shape[0]):
            if np.isclose(A_ineq[i, :] @ x_k, b_ineq[i]): working_set.append(i)
    history = []
    for k in range(50):
        g_k = G @ x_k + d; A_w_list = []
        if has_eq: A_w_list.extend(A_eq)
        if has_ineq: A_w_list.extend([A_ineq[i] for i in working_set])
        A_w = np.array(A_w_list) if A_w_list else None
        p_k, lambdas = _solve_eq_qp_subproblem(G, g_k, A_w)
        current_step = {"iteration": k, "x_k": x_k.tolist(), "g_k": g_k.tolist(), "working_set": [f"Eq {i}" for i in range(len(A_eq))] + [f"Ineq {i}" for i in working_set] if has_eq else [f"Ineq {i}" for i in working_set]}
        if p_k is None: return {"success": False, "message": "Lỗi trong bài toán con.", "history": history}
        if norm(p_k) < 1e-7:
            current_step["action"] = "Kiểm tra nhân tử Lagrange"
            if lambdas is None: return {"success": False, "message": "Không thể tính nhân tử Lagrange."}
            ineq_lambda_start_idx = A_eq.shape[0] if has_eq else 0
            ineq_lambdas = lambdas[ineq_lambda_start_idx:]
            if len(ineq_lambdas) == 0 or np.all(ineq_lambdas >= -1e-7):
                current_step["action"] += " -> Đã tối ưu."
                history.append(current_step); obj_val = 0.5 * x_k.T @ G @ x_k + d.T @ x_k
                return {"success": True, "message": f"Tìm thấy nghiệm tối ưu sau {k} bước.", "solution": x_k.tolist(), "objective_value": obj_val, "history": history}
            else:
                min_lambda_idx_local = np.argmin(ineq_lambdas); constraint_to_drop_global_idx = working_set[min_lambda_idx_local]
                current_step["action"] += f" -> Loại bỏ ràng buộc BĐT {constraint_to_drop_global_idx}."
                history.append(current_step); working_set.pop(min_lambda_idx_local)
        else:
            current_step["action"] = "Tìm bước nhảy alpha"; alpha_k = 1.0; blocking_constraint = -1
            if has_ineq:
                for i in range(A_ineq.shape[0]):
                    if i not in working_set:
                        a_i_T_p_k = A_ineq[i, :] @ p_k
                        if a_i_T_p_k < -1e-7:
                            alpha_i = (b_ineq[i] - A_ineq[i, :] @ x_k) / a_i_T_p_k
                            if alpha_i < alpha_k: alpha_k = alpha_i; blocking_constraint = i
            x_k = x_k + alpha_k * p_k
            if blocking_constraint != -1:
                current_step["action"] += f" -> Ràng buộc {blocking_constraint} chặn lại."
                working_set.append(blocking_constraint)
            else: current_step["action"] += " -> Bước đi đầy đủ."
            history.append(current_step)
    return {"success": False, "message": "Không hội tụ sau 50 bước.", "history": history}

def _solve_eq_qp_subproblem(G, g, A):
    """Hàm phụ cho Active Set & SQP: Giải bài toán con QP chỉ có ràng buộc đẳng thức."""
    if A is None or A.shape[0] == 0:
        try: p = np.linalg.solve(G, -g)
        except np.linalg.LinAlgError: p = None
        return p, None
    Z = scipy.linalg.null_space(A)
    if Z.shape[1] == 0: return np.zeros(g.shape[0]), pinv(A.T) @ g
    G_reduced = Z.T @ G @ Z; d_reduced = Z.T @ g
    try: v = np.linalg.solve(G_reduced, -d_reduced)
    except np.linalg.LinAlgError: return None, None
    p = Z @ v
    try: lambdas = pinv(A.T) @ (G @ p + g)
    except np.linalg.LinAlgError: lambdas = None
    return p, lambdas

def solve_sqp(user_code):
    """(6) NLP - SQP: Giải bài toán phi tuyến tổng quát bằng cách giải tuần tự các bài toán QP."""
    safe_globals = {'np': np, 'norm': norm, '__builtins__': {'list': list, 'dict': dict, 'tuple': tuple, 'range': range, 'len': len, 'print': print}}
    local_env = {}; exec(user_code, safe_globals, local_env)
    f, g, x0 = local_env.get('objective_func'), local_env.get('grad_func'), local_env.get('x0')
    c_eq_func, A_eq_func = local_env.get('eq_constraint_func'), local_env.get('eq_jacobian_func')
    c_ineq_func, A_ineq_func = local_env.get('ineq_constraint_func'), local_env.get('ineq_jacobian_func')
    if not all([f, g, x0]): raise ValueError("Cần định nghĩa objective_func, grad_func, và x0.")
    x_k = np.array(x0, dtype=float); n = len(x_k); B_k = np.identity(n)
    max_iter, tolerance, history = 30, 1e-5, []
    for k in range(max_iter):
        grad_k = g(x_k); G_qp = B_k; d_qp = grad_k
        A_eq_k, b_eq_k, A_ineq_k, b_ineq_k = None, None, None, None
        if c_eq_func: A_eq_k = A_eq_func(x_k); b_eq_k = -np.array(c_eq_func(x_k))
        if c_ineq_func: A_ineq_k = A_ineq_func(x_k); b_ineq_k = -np.array(c_ineq_func(x_k))
        qp_result = solve_qp_active_set(G_qp, d_qp, A_eq_k, b_eq_k, A_ineq_k, b_ineq_k, np.zeros(n))
        if not qp_result['success']: return {"success": False, "message": f"Lỗi ở bước lặp {k}: không giải được bài toán con QP.", "history": history}
        p_k = np.array(qp_result['solution'])
        history.append({"iteration": k, "x_k": x_k.tolist(), "f_x_k": f(x_k), "p_norm": norm(p_k)})
        if norm(p_k) < tolerance: return {"success": True, "message": f"Thuật toán hội tụ sau {k} bước.", "solution": x_k.tolist(), "objective_value": f(x_k), "history": history}
        alpha_k = 1.0; x_next = x_k + alpha_k * p_k
        s_k = x_next - x_k; y_k = g(x_next) - g(x_k)
        if np.dot(s_k, y_k) > 1e-7:
            Bs = B_k @ s_k; sBs = s_k.T @ Bs
            B_k = B_k - np.outer(Bs, Bs) / sBs + np.outer(y_k, y_k) / np.dot(y_k, s_k)
        x_k = x_next
    return {"success": False, "message": f"Không hội tụ sau {max_iter} bước.", "solution": x_k.tolist(), "objective_value": f(x_k), "history": history}

def solve_newton_method(user_code):
    """(7) Tối ưu không ràng buộc - Newton: Sử dụng đạo hàm bậc 2, hội tụ rất nhanh."""
    safe_globals = {'np': np, 'inv': inv, 'norm': norm, '__builtins__': {'list': list, 'dict': dict, 'tuple': tuple, 'range': range, 'len': len, 'print': print}}
    local_env = {}; exec(user_code, safe_globals, local_env)
    required_keys = ['objective_func', 'gradient_func', 'hessian_func', 'x0']
    for key in required_keys:
        if key not in local_env: raise ValueError(f"Chưa định nghĩa biến/hàm bắt buộc: '{key}'.")
    f, grad_f, hess_f = local_env['objective_func'], local_env['gradient_func'], local_env['hessian_func']
    x_k = np.array(local_env['x0'], dtype=float); max_iter, tolerance, history = 50, 1e-6, []
    for i in range(max_iter):
        f_val, g_val = f(x_k), grad_f(x_k)
        history.append({"iteration": i, "x_k": x_k.tolist(), "f_x_k": f_val, "grad_norm": norm(g_val)})
        if norm(g_val) < tolerance: return {"success": True, "message": f"Thuật toán hội tụ sau {i} bước.", "history": history, "solution": x_k.tolist(), "objective_value": f_val}
        try: p_k = -inv(hess_f(x_k)) @ g_val
        except np.linalg.LinAlgError: return {"success": False, "message": f"Lỗi ở bước lặp {i}: Ma trận Hessian không khả nghịch.", "history": history}
        x_k = x_k + p_k
    return {"success": False, "message": f"Không hội tụ sau {max_iter} bước.", "history": history, "solution": x_k.tolist(), "objective_value": f(x_k)}

def solve_gradient_descent(user_code):
    """(8) Tối ưu không ràng buộc - Gradient: Phương pháp cơ bản nhất, dùng đạo hàm bậc 1."""
    safe_globals = {'np': np, 'norm': norm, '__builtins__': {'list': list, 'dict': dict, 'tuple': tuple, 'range': range, 'len': len, 'print': print}}
    local_env = {}; exec(user_code, safe_globals, local_env)
    required_keys = ['objective_func', 'gradient_func', 'x0']
    for key in required_keys:
        if key not in local_env: raise ValueError(f"Chưa định nghĩa biến/hàm bắt buộc: '{key}'.")
    f, grad_f = local_env['objective_func'], local_env['gradient_func']
    x_k = np.array(local_env['x0'], dtype=float); max_iter, tolerance, history = 50, 1e-6, []
    for i in range(max_iter):
        f_val, g_k = f(x_k), grad_f(x_k); grad_norm = norm(g_k)
        if grad_norm < tolerance:
            history.append({"iteration": i, "x_k": x_k.tolist(), "f_x_k": f_val, "grad_norm": grad_norm, "alpha_k": None})
            return {"success": True, "message": f"Thuật toán hội tụ sau {i} bước.", "history": history, "solution": x_k.tolist(), "objective_value": f_val}
        phi = lambda alpha: f(x_k - alpha * g_k)
        res_alpha = minimize_scalar(phi, bounds=(0, 5), method='bounded'); alpha_k = res_alpha.x
        history.append({"iteration": i, "x_k": x_k.tolist(), "f_x_k": f_val, "grad_norm": grad_norm, "alpha_k": alpha_k})
        x_k = x_k - alpha_k * g_k
    history.append({"iteration": max_iter, "x_k": x_k.tolist(), "f_x_k": f(x_k), "grad_norm": norm(grad_f(x_k)), "alpha_k": None})
    return {"success": False, "message": f"Không hội tụ sau {max_iter} bước.", "history": history, "solution": x_k.tolist(), "objective_value": f(x_k)}

# ==============================================================================
# FLASK ROUTES
# ==============================================================================
@app.route('/')
def index():
    """Render trang web chính."""
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    """Endpoint chính, nhận yêu cầu và điều phối đến hàm giải phù hợp."""
    data = request.get_json()
    problem_type = data.get('type')
    
    solvers = {
        'scipy': lambda d: solve_scipy_problem(d.get('code')),
        'simplex': lambda d: solve_with_simplex_tableau(np.array(d.get('c'), dtype=float), np.array(d.get('A_eq'), dtype=float), np.array(d.get('b_eq'), dtype=float)),
        'nlp_barrier': lambda d: solve_nlp_barrier_method(d.get('code')),
        'qp_null_space': lambda d: solve_qp_null_space(d.get('G'), d.get('d'), d.get('A'), d.get('b')),
        'qp_active_set': lambda d: solve_qp_active_set(d.get('G'), d.get('d'), d.get('A_eq'), d.get('b_eq'), d.get('A_ineq'), d.get('b_ineq'), d.get('x0')),
        'sqp': lambda d: solve_sqp(d.get('code')),
        'newton': lambda d: solve_newton_method(d.get('code')),
        'gradient_descent': lambda d: solve_gradient_descent(d.get('code')),
    }

    try:
        if problem_type in solvers:
            result = solvers[problem_type](data)
        else:
            return jsonify({'error': 'Loại bài toán không hợp lệ'}), 400
        return jsonify(result)
    except Exception as e:
        error_message = f"Đã xảy ra lỗi:\n{traceback.format_exc()}"
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)