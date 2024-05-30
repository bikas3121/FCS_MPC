function [u_direct]= directQuantization(Q, ref)
    u_direct = zeros(length(ref),1);
    for i = 1:length(ref)
     err_i = zeros(length(Q),1);
     for j  = 1:length(Q)
         err_i(j) = norm(ref(i)-Q(j));
     end
        index_minerr = find(err_i == min(err_i),1);
        u_direct(i) = Q(index_minerr);
    end
end