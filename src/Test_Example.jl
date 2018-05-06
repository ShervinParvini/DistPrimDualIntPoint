# A coupled convex quadratic programming problem with 14 variables and 7 Nodes

# Add processors
addprocs(2)

# Define functions for the algorithm on all processors
include("PDIP_Functions.jl")

# Define the Nodes
N7=Node(7,3,[7,14],nothing,nothing,[],[],[],[2 -1;-1 2],[1 1]',[1 10],[1]',[1 2],[1]',[],[],[],[],[],[],[],[],[],[],[])
N6=Node(6,3,[6,13],nothing,nothing,[],[],[],[2 -1;-1 2],[1 1]',[1 9],[1]',[1 2],[1]',[],[],[],[],[],[],[],[],[],[],[])
N4=Node(4,2,[4,11],nothing,nothing,[],[],[],[2 -1;-1 2],[1 1]',[1 3],[1]',[1 2],[1]',[],[],[],[],[],[],[],[],[],[],[])
N3=Node(3,1,[3,10],nothing,nothing,[],[],[],[2 -1;-1 2],[1 1]',[1 6],[1]',[1 2],[1]',[],[],[],[],[],[],[],[],[],[],[])
N5=Node(5,3,[5,12,13,14],nothing,[N6,N7],[],[],[],[5.8 .8 .5 .9;.8 4.1 .6 .5;.5 .6 4.2 .9;.9 .5 .9 4.1],[1 1 1 1]',[1 4 2 6;2 6 4 12],[1 2]',[1 2 3 4],[1]',[],[],[],[],[],[],[],[],[],[],[])
N2=Node(2,2,[2,9,11,12],nothing,[N4,N5],[],[],[],[5.8 .8 .5 .9;.8 4.1 .6 .5;.5 .6 4.2 .9;.9 .5 .9 4.1],[1 1 1 1]',[1 3 5 2;1 4 3 7],[1 2]',[1 2 3 4],[1]',[],[],[],[],[],[],[],[],[],[],[])
N1=Node(1,1,[1,8,9,10],nothing,[N2,N3],[],[],[],[5.8 .8 .5 .9;.8 4.1 .6 .5;.5 .6 4.2 .9;.9 .5 .9 4.1],[1 1 1 1]',[1 4 5 6;1 2 3 7;3 2 5 9],[1 2 3]',[1 2 3 4],[1]',[],[],[],[],[],[],[],[],[],[],[])

# Initialization
(n,q) = (14,7)           # Number of variables and Nodes
x_0 = .01 * ones(14,1)   # Initial starting point (should be feasible w.r.t inequality constraints)
Beta = .8                # Beta for backtracking line search
alpha = 0.1              # alpha for backtracking line search
mu = 15                  # mu for updating the barrier coefficient
epsilon_feas = 1e-6      # Primal and dual feasibility threshold
epsilon = 1e-6           # Surrogate duality gap threshold

x = x_0

setparent(N1)             # Set parent Nodes

setchildrenprocsid(N1)    # Set the processor id of child Nodes in which they are defined

SepRes(N1)                # Define separators and residuals

Preprocessing_EquCons(N1) # Perform preprocessing on equality constraints (QR factorization)

(m,p) = find_m_p(N1)      # Find the number of equality and inequality constraints

init_lambda_v_x(N1,x)     # initialize x, lambda and v

############################################
# Send N2 to processor 2
subtree_to_send = Base.deepcopy(N2)
subtree_to_send.parent = nothing  # No need to provide N2 with information regarding its parent Node
sendto(2, subtree=subtree_to_send)

# Send N5 to processor 3
subtree_to_send = nothing        # Clean the previously defined subtree_to_send
subtree_to_send = Base.deepcopy(N5)
subtree_to_send.parent = nothing  # No need to provide N2 with information regarding its parent Node
sendto(3, subtree=subtree_to_send)
############################################
# initialize the algorithm parameters
iter = 1
step = [0.];
back_no = [0.];

sur_duality = find_surduality(N1)
sur_duality=[sur_duality]

(res_primal_tot,res_dual_tot,res_cent_tot) =  TotalResiduals(N1,mu*m/sur_duality[1],n)
res_primal_tot_norm = (1/p) * norm(res_primal_tot)
res_dual_tot_norm = (1/m) * norm(res_dual_tot)

###############################################
# Start iterating !

while (sur_duality[iter] > epsilon || res_primal_tot_norm > epsilon_feas || res_dual_tot_norm > epsilon_feas)

  # Update the parameter t
  t1 = mu*m/sur_duality[iter];

  # Find the search direction
  MessagePassingDirection(N1,t1,n)

  # Find the corresponding Delta_lambda
  find_Delta_lambda(N1)

  # Find a proper step size
  StepInfo = MessagePassingStepSize(N1,t1,alpha,Beta,n)
  push!(step,StepInfo[1])
  push!(back_no,StepInfo[2])

  if back_no[iter+1]==1
    println("Algorithm could not find a proper step size!")
    break
  end

  # Update x, lambda and v
  Update_lambda_v_x(N1,step[iter+1])

  # Increment the iterator
  iter += 1

  # Find the surrogate duality gap for checking the termination criteria
  push!(sur_duality,0)
  sur_duality[iter] = find_surduality(N1)

  # Find the primal and dual residuals for checking the termination criteria
  (res_primal_tot,res_dual_tot,res_cent_tot) =  TotalResiduals(N1,t1,n)
  res_primal_tot_norm = (1/p) * norm(res_primal_tot)
  res_dual_tot_norm = (1/m) * norm(res_dual_tot)
end

# Display x, lambda and v locally on each Node
display(N1)
display(N3)
display(@fetchfrom 2 subtree)
display(@fetchfrom 2 subtree.children[1])
display(@fetchfrom 3 subtree)
display(@fetchfrom 3 subtree.children[1])
display(@fetchfrom 3 subtree.children[2])

# Display the algorithm parameters
println("Iteration number is $(iter-1)")                                # number of primal-dual interior-point iterations
println("Step size at each iteration is $(step)")                       # Step sizes at each iteration
println("Surrogate duality gap at each iteration is $(sur_duality)")    # Surrogate duality gap at each iteration
