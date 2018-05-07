# Functions defined for the distributed primal-dual interior-point algorithm presented in
# the paper "Distributed primal–dual interior-point methods for solving tree-structured coupled convex problems
# using message-passing" by Sina Khoshfetrat Pakazad, Anders Hansson, Martin S. Andersen & Isak Nielsen

# (c): Shervin Parvini Ahmadi

@everywhere function sendto(p::Int; args...)   # This function is taken from the ParallelDataTransfer library on Github and is slightly modified
      for (nm, val) in args
          wait(@spawnat(p, eval(Main, Expr(:(=), nm, val))))
      end
end

@everywhere type Node
  Nodename::Int64                        # Name Nodes with Integers (useful for debugging)
  ProcsId::Int64                         # The processor in which the Node is supposed to live
  var::Vector{Int64}                     # Index of variables
  parent::Union{Node,Void}               # Parent Node
  children::Union{Vector{Node},Void}     # Child Nodes
  ChProcsId::Vector{Int64}               # The processors in which the children Nodes are supposed to live
  Sep                                    # Separator of variables w.r.t variables defined on child Nodes
  Res::Vector{Int64}                     # Residual of variables w.r.t variables defined on parent Node
  H::Array                               # Quadratic term (square matrix) in the objective function
  g::Array                               # Linear term (column vector) in the objective function
  A::Array                               # x's coefficient in the equality constraint
  b::Array                               # Constant term in the equality constraint
  A_I::Array                             # x's coefficient in the inequality constraint
  b_I::Array                             # Constant term in the inequality constraint
  x::Array                               # x
  Delta_x::Array                         # Delata_x
  lambda::Array                          # lambda (the multiplier in Lagrangian which corresponds to the inequality constraint)
  Delta_lambda::Array                    # Delta_lambda
  v::Array                               # v (the multiplier in Lagrangian which corresponds to the equality constraint)
  Delta_v::Array                         # Delta_v
  res_cent::Array                        # Needs to be stored in the upward pass stage in order to be used to calculate Delta_lambda later
  H1::Array                              # Partitioning in the upward pass stage which needs to be stored to be used in the downward-pass stage
  H2::Array                              # Partitioning in the upward pass stage which needs to be stored to be used in the downward-pass stage
  h1::Array                              # Partitioning in the upward pass stage which needs to be stored to be used in the downward-pass stage
  h2::Array                              # Partitioning in the upward pass stage which needs to be stored to be used in the downward-pass stage
end

function setparent(N::Node)
  if ~(N.children == nothing)
    for i=1:length(N.children)
      N.children[i].parent = N;
      setparent(N.children[i]);
    end
  end
end

function setchildrenprocsid(N::Node)
  if ~(N.children == nothing)
    for i=1:length(N.children)
      push!(N.children[i].parent.ChProcsId,N.children[i].ProcsId);
      setchildrenprocsid(N.children[i]);
    end
  end
end

function SepRes(N::Node)

  if N.parent == Void()
    N.Res = N.var
  else
    for i in N.var
      if isempty(find(N.parent.var.==i))
        push!(N.Res,i)
      end
    end
  end
  if ~(N.children==Void())
    for i in 1:length(N.children)
      push!(N.Sep,[]);
    end
    for i in N.var
      for j in 1:length(N.children)
        if ~isempty(find(N.children[j].var.==i))
          push!(N.Sep[j],i)
        end
      end
    end
  end

  if ~(N.children == nothing)
    for k=1:length(N.children)
      SepRes(N.children[k])
    end
  end

end

function Preprocessing_EquCons(N::Node)
  E_RC=zeros(length(N.Res),length(N.var)); for iii=1:length(N.var); if any(N.Res.==N.var[iii]);E_RC[find(N.Res.==N.var[iii]),iii]=1;end;end
  if ~(N.children == nothing)
    for k=1:length(N.children)
      received_message =  Preprocessing_EquCons(N.children[k])
      if ~(received_message == nothing)
        E_SC=zeros(length(N.Sep[k]),length(N.var)); for iii=1:length(N.var); if any(N.Sep[k].==N.var[iii]);E_SC[find(N.Sep[k].==N.var[iii]),iii]=1;end;end
        N.A = [N.A; received_message[1] * E_SC];
        N.b = [N.b; received_message[2]];
      end
    end
  end
  if rank(N.A * E_RC') < size(N.A,1)
    if ~(N.parent == nothing)
      E_SC=zeros(length(setdiff(N.var,N.Res)),length(N.var)); for iii=1:length(N.var); if any(setdiff(N.var,N.Res).==N.var[iii]);E_SC[find(setdiff(N.var,N.Res).==N.var[iii]),iii]=1;end;end
      QR = qrfact(N.A * E_RC');
      trans_A = QR[:Q]'*[N.A*E_RC' N.A*E_SC'];
      trans_b = QR[:Q]' * N.b;
      msg_A = trans_A[rank(N.A * E_RC')+1:end,length(N.Res)+1:end];
      msg_b = trans_b[rank(N.A * E_RC')+1:end,:];
      rank_A_RC = rank(N.A * E_RC')
      N.A = trans_A[1:rank_A_RC,:] * [E_RC;E_SC];
      N.b = trans_b[1:rank_A_RC,:];
      return(msg_A,msg_b)
    else
      QR = qrfact(N.A);
      trans_A = QR[:Q]' * N.A;
      trans_b = QR[:Q]' * N.b;
      rank_A = rank(N.A)
      N.A = trans_A[1:rank_A,:];
      N.b = trans_b[1:rank_A,:];
      return(nothing)
    end
  else
    return(nothing)
  end
end

function find_m_p(N::Node)
  m=size(N.A_I,1);
  p=size(N.A,1);
  if ~(N.children == nothing)
    for i=1:length(N.children)
      m_p = find_m_p(N.children[i])
      m += m_p[1]; p += m_p[2];
    end
  end
  return(m,p)
end

@everywhere function init_lambda_v_x(N::Node,x)
  N.lambda = ones(size(N.A_I,1),1)
  N.v = ones(size(N.A,1),1)
  N.x = x[N.var]
  N.Delta_x = Matrix(length(N.var),1)
  if ~(N.children == nothing)
    for i=1:length(N.children)
      init_lambda_v_x(N.children[i],x)
    end
  end
end

@everywhere function ResidualPDMessagePassing(N::Node,x_loc,lambda,v,t1)
  res_primal = N.A * x_loc - N.b
  if ~isempty(N.A_I)
    res_cent = -diagm(vec(lambda)) * (N.A_I * x_loc - N.b_I) - 1/t1*ones(length(lambda),1)
    res_dual = N.H*x_loc + N.g + N.A_I'*lambda + N.A'*v
  else
    res_dual =  N.H*x_loc + N.g  + N.A'*v
    res_cent = [];
  end

  return(res_primal,res_dual,res_cent)
end

@everywhere function upwardpass(N::Node,t1)
  E_RC=zeros(length(N.Res),length(N.var)); for iii=1:length(N.var); if any(N.Res.==N.var[iii]);E_RC[find(N.Res.==N.var[iii]),iii]=1;end;end
  (res_primal,res_dual,N.res_cent) = ResidualPDMessagePassing(N,N.x,N.lambda,N.v,t1)
  H_msg = N.H
  if ~isempty(N.A_I)
    for j=1:size(N.A_I,1)
      H_msg -= (N.lambda[j]./(N.A_I[j,:]'*N.x-N.b_I[j])).*N.A_I[j,:]*N.A_I[j,:]'
    end
    r_msg = res_dual + N.A_I'*diagm(vec(1./(N.A_I*N.x-N.b_I)))*N.res_cent
  else
    r_msg = res_dual
  end

  if ~(N.children == nothing)

      if any(N.ChProcsId.!=myid())
        abroad_children_indices = find(N.ChProcsId.!=myid())
        msg_Q2_q2 = Vector{Future}(length(abroad_children_indices))
        @sync for (idx, pid) in enumerate(N.ChProcsId[abroad_children_indices])
          @async msg_Q2_q2[idx] = @spawnat pid upwardpass_wraptree(t1)
        end

        if any(N.ChProcsId.==myid())
          for i in find(N.ChProcsId.==myid())
            E_SC=zeros(length(N.Sep[i]),length(N.var)); for iii=1:length(N.var); if any(N.Sep[i].==N.var[iii]);E_SC[find(N.Sep[i].==N.var[iii]),iii]=1;end;end ###
            (msg_Q,msg_q) = upwardpass(N.children[i],t1)
            H_msg += E_SC' * msg_Q * E_SC;
            r_msg += E_SC' * msg_q;
          end
        end

        for (idx,i) in enumerate(abroad_children_indices)
          (msg_Q2,msg_q2) = fetch(msg_Q2_q2[idx])
          E_SC=zeros(length(N.Sep[i]),length(N.var)); for iii=1:length(N.var); if any(N.Sep[i].==N.var[iii]);E_SC[find(N.Sep[i].==N.var[iii]),iii]=1;end;end
          H_msg += E_SC' * msg_Q2 * E_SC;
          r_msg += E_SC' * msg_q2;
        end

    else
      for i=1:length(N.children)
        E_SC=zeros(length(N.Sep[i]),length(N.var)); for iii=1:length(N.var); if any(N.Sep[i].==N.var[iii]);E_SC[find(N.Sep[i].==N.var[iii]),iii]=1;end;end
        (msg_Q,msg_q) = upwardpass(N.children[i],t1)
        H_msg += E_SC' * msg_Q * E_SC;
        r_msg += E_SC' * msg_q;
      end
    end
  end

  E_SC=zeros(length(setdiff(N.var,N.Res)),length(N.var)); for iii=1:length(N.var); if any(setdiff(N.var,N.Res).==N.var[iii]);E_SC[find(setdiff(N.var,N.Res).==N.var[iii]),iii]=1;end;end

  Q_yy = E_SC * H_msg * E_SC';
  q_y = E_SC * r_msg;
  A_y = N.A * E_SC';
  Q_zy= E_RC * H_msg * E_SC';
  Q_zz = E_RC * H_msg * E_RC';
  q_z = E_RC * r_msg;
  A_z = N.A * E_RC';

  O = [Q_zz A_z';A_z zeros(size(A_z,1),size(A_z,1))];

  H_opt = -O \ [Q_zy;A_y];
  N.H1 = H_opt[1:length(N.Res),:];
  N.H2 = H_opt[length(N.Res)+1:end,:];

  h_opt = -O \ [q_z;res_primal];
  N.h1 = h_opt[1:length(N.Res),:];
  N.h2 = h_opt[length(N.Res)+1:end,:];

  msg_Q = N.H1' * Q_zz * N.H1 + N.H1' * Q_zy + Q_zy' * N.H1 + Q_yy;
  msg_q = N.H1' * Q_zz * N.h1 + N.H1' * q_z + Q_zy' * N.h1 + q_y;
  return(msg_Q,msg_q)

end

@everywhere function upwardpass_wraptree(t1)
  get_true_tree = subtree
  return(upwardpass(get_true_tree,t1))
end

@everywhere function downwardpass(N::Node,Delta_x_send)

  if setdiff(N.var,N.Res)==[]
    N.Delta_x =  N.h1
    N.Delta_v =  N.h2
  else
    N.Delta_x[findin(N.var,N.Res)] = N.H1 * Delta_x_send + N.h1
    N.Delta_x[findin(N.var,setdiff(N.var,N.Res))] = Delta_x_send
    N.Delta_v = N.H2 * Delta_x_send + N.h2
  end

  if ~(N.children == nothing)
    if any(N.ChProcsId.!=myid())
      abroad_children_indices = find(N.ChProcsId.!=myid())
      confirm = Vector{Future}(length(abroad_children_indices))
      for (idx, pid) in enumerate(N.ChProcsId[abroad_children_indices])
        to_send1 = N.Delta_x[findin(N.var,N.Sep[abroad_children_indices[idx]])]
		confirm[idx] = @spawnat pid downwardpass_wraptree(to_send1)
      end

      if any(N.ChProcsId.==myid())
        for i in find(N.ChProcsId.==myid())
          to_send2 = N.Delta_x[findin(N.var,N.Sep[i])]
		  downwardpass(N.children[i],to_send2)
        end
      end
      @sync for idx = 1:length(abroad_children_indices)
        @async wait(confirm[idx])
      end

    else
      for i=1:length(N.children)
        to_send3 = N.Delta_x[findin(N.var,N.Sep[i])]
		downwardpass(N.children[i],to_send3)
      end
    end
  end
end

@everywhere function downwardpass_wraptree(Delta_x_send)
  get_true_tree = subtree
  return(downwardpass(get_true_tree,Delta_x_send))
end

function MessagePassingDirection(N::Node,t1,n)
  upwardpass(N,t1)
  Delta_x_tot = downwardpass(N,nothing)
end

@everywhere function  find_Delta_lambda(N::Node)
  N.Delta_lambda = -diagm(vec(1./(N.A_I*N.x-N.b_I)))*(diagm(vec(N.lambda))*N.A_I*N.Delta_x - N.res_cent)
  if ~(N.children == nothing)
    if any(N.ChProcsId.!=myid())
      abroad_children_indices = find(N.ChProcsId.!=myid())
      confirm = Vector{Future}(length(abroad_children_indices))
      @sync for (idx, pid) in enumerate(N.ChProcsId[abroad_children_indices])
          @async confirm[idx] = @spawnat pid find_Delta_lambda_wraptree()
      end
      if any(N.ChProcsId.==myid())
        for i in find(N.ChProcsId.==myid())
          find_Delta_lambda(N.children[i])
        end
      end
      @sync for idx = 1:length(abroad_children_indices)
        @async wait(confirm[idx])
      end

    else
      for i=1:length(N.children)
        find_Delta_lambda(N.children[i])
      end
    end
  end
  return(nothing)
end

@everywhere function find_Delta_lambda_wraptree()
  get_true_tree = subtree
  return(find_Delta_lambda(get_true_tree))
end

@everywhere function stepsize_firststage(N::Node,Beta)
  Temp = -N.lambda./N.Delta_lambda
  t_d = 1
  if ~isempty(Temp[N.Delta_lambda.<0])
    t_d = min(1,minimum(Temp[N.Delta_lambda.<0]))
  end
  t = 0.99 * t_d
  while sum(N.A_I*(N.x+t*N.Delta_x)-N.b_I.>=0)>0
    t = Beta * t;
  end

  if ~(N.children == nothing)
    if any(N.ChProcsId.!=myid())
      abroad_children_indices = find(N.ChProcsId.!=myid())
      t_msg = Vector{Future}(length(abroad_children_indices))
      @sync for (idx, pid) in enumerate(N.ChProcsId[abroad_children_indices])
        @async t_msg[idx] = @spawnat pid stepsize_firststage_wraptree(Beta)
      end
      if any(N.ChProcsId.==myid())
        for i in find(N.ChProcsId.==myid())
          t = min(t,stepsize_firststage(N.children[i],Beta))
        end
      end
      for idx=1:length(abroad_children_indices)
        t = min(t,fetch(t_msg[idx]))
      end

    else
      for i=1:length(N.children)
        t = min(t,stepsize_firststage(N.children[i],Beta))
      end
    end

  end
  return(t)
end

@everywhere function stepsize_firststage_wraptree(Beta)
  get_true_tree = subtree
  return(stepsize_firststage(get_true_tree,Beta))
end

@everywhere function TotalResiduals(N::Node,t1,n)
  (res_primal_tot,res_dual1,res_cent_tot) = ResidualPDMessagePassing(N,N.x,N.lambda,N.v,t1)
  res_dual_tot=zeros(n,1);
  kk=1;for ii in N.var; res_dual_tot[ii,1]+=res_dual1[kk,1];kk+=1;end
  if ~(N.children == nothing)
    if any(N.ChProcsId.!=myid())
      abroad_children_indices = find(N.ChProcsId.!=myid())
      res_msg_reciept = Vector{Future}(length(abroad_children_indices))
      @sync for (idx, pid) in enumerate(N.ChProcsId[abroad_children_indices])
        @async res_msg_reciept[idx] = @spawnat pid TotalResiduals_wraptree(t1,n)
      end
      if any(N.ChProcsId.==myid())
        for i in find(N.ChProcsId.==myid())
          (res_primal,res_dual2,res_cent) = TotalResiduals(N.children[i],t1,n)
          res_primal_tot = [res_primal_tot;res_primal]
          res_dual_tot += res_dual2
          res_cent_tot = [res_cent_tot;res_cent]
        end
      end
      for idx=1:length(abroad_children_indices)
        res_msg = fetch(res_msg_reciept[idx])
        res_primal_tot = [res_primal_tot;res_msg[1]]
        res_dual_tot += res_msg[2]
        res_cent_tot = [res_cent_tot;res_msg[3]]
      end

    else
      for i=1:length(N.children)
        (res_primal,res_dual2,res_cent) = TotalResiduals(N.children[i],t1,n)
        res_primal_tot = [res_primal_tot;res_primal]
        res_dual_tot += res_dual2
        res_cent_tot = [res_cent_tot;res_cent]
      end
    end
  end
  return(res_primal_tot,res_dual_tot,res_cent_tot)
end

@everywhere function TotalResiduals_wraptree(t1,n)
  get_true_tree = subtree
  return(TotalResiduals(get_true_tree,t1,n))
end

@everywhere function TotalResiduals_Ahead(N::Node,t1,n,t_step)
  (res_primal_tot,res_dual1,res_cent_tot) = ResidualPDMessagePassing(N,N.x+t_step*N.Delta_x,N.lambda+t_step*N.Delta_lambda,N.v+t_step*N.Delta_v,t1)
  res_dual_tot=zeros(n,1);
  kk=1;for ii in N.var; res_dual_tot[ii,1]+=res_dual1[kk,1];kk+=1;end
  if ~(N.children == nothing)
    if any(N.ChProcsId.!=myid())
      abroad_children_indices = find(N.ChProcsId.!=myid())
      res_msg_reciept = Vector{Future}(length(abroad_children_indices))
      @sync for (idx, pid) in enumerate(N.ChProcsId[abroad_children_indices])
        @async res_msg_reciept[idx] = @spawnat pid TotalResiduals_Ahead_wraptree(t1,n,t_step)
      end
      if any(N.ChProcsId.==myid())
        for i in find(N.ChProcsId.==myid())
          (res_primal,res_dual2,res_cent) = TotalResiduals_Ahead(N.children[i],t1,n,t_step)
          res_primal_tot = [res_primal_tot;res_primal]
          res_dual_tot += res_dual2
          res_cent_tot = [res_cent_tot;res_cent]
        end
      end
      for idx=1:length(abroad_children_indices)
        res_msg = fetch(res_msg_reciept[idx])
        res_primal_tot = [res_primal_tot;res_msg[1]]
        res_dual_tot += res_msg[2]
        res_cent_tot = [res_cent_tot;res_msg[3]]
      end
    else
      for i=1:length(N.children)
        (res_primal,res_dual2,res_cent) = TotalResiduals_Ahead(N.children[i],t1,n,t_step)
        res_primal_tot = [res_primal_tot;res_primal]
        res_dual_tot += res_dual2
        res_cent_tot = [res_cent_tot;res_cent]
      end
    end

  end
  return(res_primal_tot,res_dual_tot,res_cent_tot)
end

@everywhere function TotalResiduals_Ahead_wraptree(t1,n,t_step)
  get_true_tree = subtree
  return(TotalResiduals_Ahead(get_true_tree,t1,n,t_step))
end

function MessagePassingStepSize(N::Node,t1,alpha,Beta,n)
  t_final = stepsize_firststage(N,Beta)

  residual1 = TotalResiduals(N,t1,n)
  res = norm([residual1[1];residual1[2];residual1[3]])^2
  residual2 = TotalResiduals_Ahead(N,t1,n,t_final)
  res_t = norm([residual2[1];residual2[2];residual2[3]])^2

  back_no = 0;
  if (res_t > (1-alpha*t_final)^2*res)
    back_no = 1;
  end
  return(t_final,back_no)
end

@everywhere function find_surduality(N::Node)
  sur_duality = -( (N.A_I*N.x - N.b_I)'*N.lambda );
  if ~(N.children == nothing)
    if any(N.ChProcsId.!=myid())
      abroad_children_indices = find(N.ChProcsId.!=myid())
      sur_duality_msg_reciept = Vector{Future}(length(abroad_children_indices))
      @sync for (idx, pid) in enumerate(N.ChProcsId[abroad_children_indices])
        @async sur_duality_msg_reciept[idx] = @spawnat pid find_surduality_wraptree()
      end
      if any(N.ChProcsId.==myid())
        for i in find(N.ChProcsId.==myid())
          sur_duality += find_surduality(N.children[i])
        end
      end
      for idx=1:length(abroad_children_indices)
        sur_duality += fetch(sur_duality_msg_reciept[idx])
      end

    else
      for i=1:length(N.children)
        sur_duality += find_surduality(N.children[i])
      end
    end

  end
  sur_duality = reshape(sur_duality,1)[1]
  return(sur_duality)
end

@everywhere function find_surduality_wraptree()
  get_true_tree = subtree
  return(find_surduality(get_true_tree))
end

@everywhere function Update_lambda_v_x(N::Node,stepsize)
  N.lambda += stepsize * N.Delta_lambda
  N.v += stepsize * N.Delta_v
  N.x += stepsize * N.Delta_x
  if ~(N.children == nothing)
    if any(N.ChProcsId.!=myid())
      abroad_children_indices = find(N.ChProcsId.!=myid())
      confirm = Vector{Future}(length(abroad_children_indices))
      @sync for (idx, pid) in enumerate(N.ChProcsId[abroad_children_indices])
        @async confirm[idx] = @spawnat pid Update_lambda_v_x_wraptree(stepsize)
      end
      if any(N.ChProcsId.==myid())
        for i in find(N.ChProcsId.==myid())
          Update_lambda_v_x(N.children[i],stepsize)
        end
      end
      @sync for idx = 1:length(abroad_children_indices)
        @async wait(confirm[idx])
      end

    else
      for i=1:length(N.children)
        Update_lambda_v_x(N.children[i],stepsize)
      end
    end

  end
  return(nothing)
end

@everywhere function Update_lambda_v_x_wraptree(stepsize)
  get_true_tree = subtree
  return(Update_lambda_v_x(get_true_tree,stepsize))
end
