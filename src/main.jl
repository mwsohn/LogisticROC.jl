"""
    rocinput(logitmodel)

Takes a logit model from a GLM regression and produces two vectors `target` and `non-target`
for input to `roc` function in the ROCAnalysis package.
"""
function rocinput(logitmodel)
    res = response(logitmodel)
    pred = predict(logitmodel)
    mask = res .== 1.0
    return (pred[mask], pred[.!mask])
end

import Plots.plot
"""
    plot(::Roc)

Produces an ROC Curve plot from a `Roc` object. `Roc` struct is defined in the ROCAnalysis package and
is created by `roc` function in the same package.
"""
function plot(rocdata::Roc)
    plt = plot(rocdata.pfa, 1.0 .- rocdata.pmiss, linetype=:steppost, legend=false, xlabel="1 - specificity", ylabel="sensitivity")
    plot!(plt, collect(0:1:1), collect(0:1:1), legend=false, lc=:black)
    return plt
end

"""
    rocplot(logitmodel)

Produces an ROC plot using Plots.jl. It returns the `plot` object that can be used to modify the ROC plot.
"""
function rocplot(glmout)
    return plot(ROCAnalysis.roc(rocinput(glmout)...))
end

"""
    roccomp(logitmodel1, logitmmodel2)

Compares the ROC curves fro two logit models using `delong_test` function in the ROCAnalysis package.
"""
function roccomp(glmout1, glmout2)

    (tar1, nontar1) = rocinput(glmout1)
    (tar2, nontar2) = rocinput(glmout2)

    return ROCAnalysis.delong_test(tar1, nontar1, tar2, nontar2)
end

"""
    lroc(logitmodel; rocplot = true)

Prints the Area under the ROC Curve for a logit model and creates a ROC Curve plot. If you don't want the plot,
set `rocplot` option to `false`.
"""
function lroc(glmout; rocplot=true)
    rr = ROCAnalysis.roc(rocinput(glmout)...)
    println("Number of observations = ", nobs(glmout))
    rocval = ROCAnalysis.AUC(rr)
    println("Area under ROC curve   = ", rocval)
    if rocplot
        plt = plot(rr)
        return plt
    end
    return rocval
end

"""
    auc(logitmodel)

Returns the Area under the ROC Curve for a logit model. If you just need the AUC value, use this function.
"""
function auc(glmout)
    return ROCAnalysis.AUC(ROCAnalysis.roc(rocinput(glmout)...))
end

