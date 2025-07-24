function discreteSol = discretizeSolution(sol, T_num, prevSol)
    % sol: sürekli çözüm (vektör)
    % T_num: transfer fonksiyonu tipi (1–8)
    % prevSol: bir önceki binary çözüm (V-shaped fonksiyonlarda gerekli)
    
    D = length(sol);
    discreteSol = zeros(1, D);
    
    if isempty(prevSol) && T_num >= 5
        prevSol = randi([0 1], 1, D);  % rastgele 0/1 vektörü
    end

    for j = 1:D
        v = sol(j); % Velocity yerine burada 'sürekli çözüm' değeri kullanılıyor

        % Transfer fonksiyonu
        switch T_num
            case 1  % S1
                s = 1 / (1 + exp(-2*v));
            case 2  % S2
                s = 1 / (1 + exp(-v));
            case 3  % S3
                s = 1 / (1 + exp(-v/2));
            case 4  % S4
                s = 1 / (1 + exp(-v/3));
            case 5  % V1
                s = abs(erf((sqrt(pi)/2)*v));
            case 6  % V2
                s = abs(tanh(v));
            case 7  % V3
                s = abs(v / sqrt(1 + v^2));
            case 8  % V4
                s = abs((2/pi) * atan((pi/2)*v));
            otherwise
                error('Invalid T_num');
        end

        % Eşikleme / karar
        if T_num >= 1 && T_num <= 4
            % S-shaped → doğrudan 0/1 ata
            if rand < s
                discreteSol(j) = 1;
            else
                discreteSol(j) = 0;
            end
        elseif T_num >= 5 && T_num <= 8
            % V-shaped → önceki çözüm üzerinden flip yap
            if rand < s
                discreteSol(j) = ~prevSol(j);  % Bit flip
            else
                discreteSol(j) = prevSol(j);   % Değiştirme
            end
        end
    end
end