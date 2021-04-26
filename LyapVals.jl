using Distributed
# Usage: julia LyapVals.jl <pathToDataFile> <outputMask> <outputDir>
addprocs(8)
@everywhere import Pkg
@everywhere Pkg.add("PyCall")
@everywhere using PyCall
@everywhere np =  pyimport("numpy")
@everywhere heteroclinicsData = np.loadtxt($(ARGS[1]))
@everywhere function prepareData(dataToPrep)
    data = Vector[dataToPrep[1,:]]
    for i in 2:size(dataToPrep[:,1])[1]
                    data=hcat(data,[dataToPrep[i,:]])
    end
    data
end
@everywhere data = prepareData(heteroclinicsData)
@everywhere include("ForLyapunovVal.jl")
@everywhere using Main.ReducedSysLyapVals

using Dates
timeOfRun = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
nameOutputFile = ARGS[2]
pathToOutputDir = ARGS[3]
outputFileMask = string(nameOutputFile, timeOfRun)
OutputFile = string(pathToOutputDir, outputFileMask )
@time begin
    result = pmap(getLyapunovData,  data)
end

if !isempty(result)
        headerStr = (
                "i  j  alpha  beta  r  LyapunovVal\n0  1  2      3     4  5")
        fmtList = ["%2u",
                   "%2u",
                   "%+18.15f",
                   "%+18.15f",
                   "%+18.15f",
                   "%+18.15f",]
        np.savetxt(OutputFile, result[1,:], header=headerStr,
                   fmt=fmtList)
end
