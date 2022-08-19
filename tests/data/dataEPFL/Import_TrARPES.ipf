#pragma rtGlobals=1		// Use modern global access method.

// re-initialize path
//newPath/O/Z/Q artemis_procedures, ":User procedures:Artemis:Igor Procedures:"

	//Michele igor7 mod
	string userpath
	userpath= SpecialDirPath("Igor Pro User Files",0,0,0)
	//print userpath
	string ARTEMIS_PATH_LINK=userpath+"User Procedures:Artemis"
	//print i2ppe_PATH_LINK
	newpath/O ARTEMIS_PATH_LINK_TMP, ARTEMIS_PATH_LINK
	pathinfo ARTEMIS_PATH_LINK_TMP//ok now S_path should have the correct link location...
	print S_path
	NewPath/O/Q artemis_procedures, S_Path+"Igor Procedures"




//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function TrARPES_Data()
	String ctrlName

	string DF = getDataFolder(1)

	// re-initialize path
	
	//also here need to edit to get the right path
	
	///////////////////////////igor 7 michele mod
	string userpath
	userpath= SpecialDirPath("Igor Pro User Files",0,0,0)
	//print userpath
	string ARTEMIS_PATH_LINK=userpath+"User Procedures:Artemis"
	//print i2ppe_PATH_LINK
	newpath/O ARTEMIS_PATH_LINK_TMP, ARTEMIS_PATH_LINK
	pathinfo ARTEMIS_PATH_LINK_TMP//ok now S_path should have the correct link location...
	print S_path
	NewPath/O/Q artemis_procedures, S_Path+"Igor Procedures"
	
////////////////////////
	
	//newPath/O/Z/Q artemis_procedures, ":User procedures:Artemis:Igor Procedures:"

	NewDataFolder/O root:Data
	NewDataFolder/O root:Data:tmpData
	NewDataFolder/O/S root:Data:tmpData:Analyser
//	For the angular correction
	Variable/G PixelSize=0.00645, magnification=4.54, WF=4.2  // 03.07.14: Claude : adapted
	Variable/G LensMode, Ek, Ep//, ErangeLow, ErangeHigh
	Variable/G E_Offset_px, Ang_Offset_px, Ang_Centre_px, Binning=4, De1, aInner
	Variable/G EkinLow, EkinHigh, AzimuthLow, AzimuthHigh
	//Claude: 22.11.13
	Variable/G Rotation_Angle=0
//	For the edge correction
	Variable/G Edge_pos, Edge_Slope
	Variable/G Edge_correction
//	For the smoothing
	Variable/G Check_2D_Smooth
	Variable/G n_size, passes
//	For the Crop
	Variable/G Crop
	Variable/G E1, E2, Theta1, Theta2
	
	// for Fourier Filtering
	Variable/G filter_Image = 0
	Variable/G filter_fx = 0.0793478
	Variable/G filter_fy = 0.076087
	Variable/G filter_wx = 0.0152
	Variable/G filter_wy = 0.0134
	Variable/G filter_A = 0.95
	
	// for setN
	Variable/G setN = NaN
	
	// for the Path
	String/G gs_trARPES_load_path = "C:Users:Michele:Documents:Nightscans:04 April:Day 29:Raw Data:"
	
	// Data Format
	Variable/G gv_DataFormat = 2 // New Format
	
	// Keep Raw Data
	Variable/G gv_keepRawData = 0
	
	// Conversion factor for CCD counts
	Variable/G CCDcounts2ecounts = 1

	if( WaveExists(Angular_Correction)!=1)
		Make/O/N=1 Angular_Correction
	endif
	DefaultParametersFrom_Calib2D("")
	Binning=4
	DoWindow/F Import_TrARPES_Data
	if (v_flag == 0)
		Execute "Import_TrARPES_Data()"
		DoWindow/T Import_TrARPES_Data, "Import Tr-ARPES Data"
	endif
	
	SetDataFolder $DF

end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Window Import_TrARPES_Data() : Panel
	PauseUpdate; Silent 1		// building window...
	NewPanel /W=(61,186,444,286) as "Import Tr-ARPES Data"
	SetVariable setRunNb,pos={23,24},size={111,16},bodyWidth=60,title="Run Nb:"
	SetVariable setRunNb,fStyle=1,limits={1,inf,0},value= _NUM:1
	SetVariable setN,pos={142,24},size={118,16},bodyWidth=60,title="N Cycles:"
	SetVariable setN,fStyle=1,limits={1,inf,0},variable= root:Data:tmpData:Analyser:setN, value= _NUM:nan
	Button buttonNewData,pos={289,22},size={70,20},proc=ButtonNewData,title="Load Data"
	Button buttonNewData,fStyle=1
	SetVariable setPath,pos={106,68},size={176,16},bodyWidth=142,title="Path:"
	SetVariable setPath,fStyle=1,value=root:Data:tmpData:Analyser:gs_trARPES_load_path
	Button buttonChoosePath,pos={289,66},size={70,20},proc=ButtonChoosePath,title="Choose..."
	Button buttonChoosePath,fStyle=1
	Button button_options,pos={24,66},size={80,20},proc=ImportData_Settings,title="Settings ..."
	Button button_options,fStyle=1
	PopupMenu popup_DataFormat,pos={232,44},size={126,21},fStyle=1, bodyWidth=130, proc=popupMenu_DataFormat
	PopupMenu popup_DataFormat,mode=1,popvalue="New trARPES Control",value= #"\"Old trARPES Control;New trARPES Control\""
	CheckBox checkbox_keepRawData,pos={124,48},size={95,14},title="keep RAW data"
	CheckBox checkbox_keepRawData,variable= root:Data:tmpData:Analyser:gv_keepRawData 
	GroupBox group0,pos={11,10},size={360,88},fStyle=1
EndMacro


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function popupMenu_DataFormat(ctrlName,popNum,popStr) : PopupMenuControl
	String ctrlName
	Variable popNum
	String popStr	
	
	NVAR dataFormat = root:Data:tmpData:Analyser:gv_DataFormat
	dataFormat = popNum
end
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function ButtonNewData(ctrlName) : ButtonControl
	String ctrlName
	ControlInfo setRunNb
	variable RunNumber = V_value
	NVAR dataFormat = root:Data:tmpData:Analyser:gv_dataFormat
	switch(dataFormat)
		case 1:
			loadOldData("", RunNumber)
			break
		case 2:
			loadNewData("", RunNumber)
			break	
		default:
			abort "Data Format not recognized"
	endswitch		
end
	
function/S loadNewData(Path, Scan, [Cycles])
	string Path
	Variable Scan
	string Cycles
	
	Variable passEnergy, kineticEnergy, LensMode, binning
	
	NVAR keepRawData = root:Data:tmpData:Analyser:gv_keepRawData
	
	if (cmpstr(Path, "")!=0)
		// Path given, overwrite old
		ButtonChoosePath("", Path=Path)
	else
		// Path not given, look at Variable and set it to it. This will open a Prompt, if it is not set
		SVAR trARPES_load_path = root:Data:tmpData:Analyser:gs_trARPES_load_path
		ButtonChoosePath("", Path=trARPES_load_path)
	endif
	if (paramIsDefault(Cycles))
		Cycles=""
	endif
	// check the path
	PathInfo/S All_Runs_Path
	if (V_flag==0)
		// path does not exist, Run fuction to get it
		ButtonChoosePath("")
	endif
	string foldername
	sprintf foldername, "%04.0f", Scan
	GetFileFolderInfo/Z/Q/P=All_Runs_Path foldername
	if (V_Flag!=0)
		abort "Scan not found!"
	endif
	NewPath/O/Z/Q CurrenRunPath, S_Path
	NewPath/O/Z/Q CurrentAveragesPath, S_Path + "AVG"
	NewPath/O/Z/Q CurrentRawPath, S_Path + "RAW"
	// get info.txt
	string filename = "info.txt"
	GetFileFolderInfo/Z/Q/P=CurrenRunPath filename
	if (V_Flag!=0)
		abort "info.txt not found."
	endif
	string ScanInfo = parseInfoTxt(S_Path)	
	// get lens mode
	string s_lensmode = StringByKey("LensMode", ScanInfo, "=", "\r")
	if (cmpstr(s_lensmode, "")==0)
		s_lensmode = StringByKey("Mode", ScanInfo, "=", "\r")
	endif
	strswitch(s_lensmode)	
		case "LowAngularDispersion":
			LensMode=0
			break
		case "WideAngleMode":
			LensMode=3
			break
		default:
			abort "Lens mode " + StringByKey("LensMode", ScanInfo, "=", "\r") + " not implemented"
	endswitch
	binning = 2^NumberByKey("Binning", ScanInfo, "=", "\r")
	nvar old_binning=root:Data:tmpData:Analyser:binning
	if (binning!=old_binning)
		// set angular center pixel
		NVAR Ang_Centre_px = root:Data:tmpData:Analyser:Ang_Centre_px
		Ang_Centre_px = 504/binning
		old_binning = binning
	endif
	DoWindow Gr_ImportData_Setting // Check if Window exists
	if (V_flag == 1)
		PopupMenu Popup_LensMode win=Gr_ImportData_Setting, mode=LensMode+1
		popupMenu_Binning("", 0, "", set_binning=binning)
	endif
	//Pass energy
	passEnergy = NumberByKey("PassEnergy", ScanInfo, "=", "\r")
	kineticEnergy = NumberByKey("KineticEnergy", ScanInfo, "=", "\r")
	string wname
	variable scansteps
	variable i,j
	string DF = getDataFolder(1)
	// What kind of scan?
	strswitch(StringByKey("ScanType", ScanInfo, "=", "\r"))
		case "delay":
			// delay scan
			// create datafolder for compatibility
			string datafolder
			variable repetitions = NumberByKey("Repetitions", ScanInfo, "=", "\r")
			if (numtype(repetitions) !=0) 
				repetitions = 0
			endif			
			sprintf datafolder, "root:Data:R%03.0f_N%d", scan, repetitions
			NewDataFolder/O/S $datafolder
			// load scanvector
			filename = "scanvector.txt"
			GetFileFolderInfo/Z/Q/P=CurrenRunPath filename
			if (V_Flag!=0)
				abort "scanvector.txt not found!"
			endif
			LoadWave/Q/J/D/W/O/K=0/N=temp/P=CurrenRunPath filename
			//sprintf wname, "Data%03.0f_t", Scan
			// compatibility...
			wname="Data_t"
			wave temp0
			duplicate/o temp0, $wname
			wave w_scanvec = $wname
			scansteps = numpnts(w_scanvec)
			// load first image
			filename = "000.tsv"
			GetFileFolderInfo/Z/Q/P=CurrentAveragesPath filename
			if (V_Flag!=0)
				abort "Scan seems to be incomplete. File " + filename + " not found."
			endif
			LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
			wave temp0
			wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
			//sprintf wname, "Data%03.0f", Scan
			// compatibility...
			wname="Data"
			// save settings of Analysis Tab
			string Settings = ""
			if (waveexists($wname))
				string notestr = note($wname)
				settings += "ThetaPosition=" + StringByKey("ThetaPosition", notestr, "=", "\r") + "\r"
				settings += "ThetaWidth=" + StringByKey("ThetaWidth", notestr, "=", "\r") + "\r"
				settings += "EPosition=" + StringByKey("EPosition", notestr, "=", "\r") + "\r"
				settings += "EWidth=" + StringByKey("EWidth", notestr, "=", "\r") + "\r"
			endif
			make/o/n=(dimsize(w_conv, 0), dimsize(w_conv,1), scansteps) $wname=0
			wave w = $wname
			setscale/P x, dimoffset(w_conv, 0), dimdelta(w_conv, 0), waveunits(w_conv, 0), w 
			setscale/P y, dimoffset(w_conv, 1), dimdelta(w_conv, 1), waveunits(w_conv, 1), w 
			utils_progressDlg(message="Delay", done=0, numDone=0, numTotal=scansteps, title="Loading Scan...")
			for (i=0; i<scansteps; i+=1)
				if (utils_progressDlg(message="Delay " + num2str(i), done=0, numDone=i, numTotal=scansteps, title="Loading Scan...")) 
					break
				endif
				if (cmpstr(cycles, "") == 0)
					// load averages, as no Cycle selection has been done
					sprintf filename, "%03.0f.tsv", i
					GetFileFolderInfo/Z/Q/P=CurrentAveragesPath filename
					if (V_Flag!=0)
						abort "Scan seems to be incomplete. File " + filename + " not found."
					endif
					LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
					wave temp0
					duplicate/o temp0, w_raw
				else
					// load RAW data, and select requested cycles
					for (j=0; j<itemsInList(cycles); j+=1)
						if (utils_progressDlg(message="Cycle " + num2str(str2num(StringFromList(j, cycles))), done=0, numDone=j, numTotal=itemsInList(cycles), title="Loading Scan...", level=1)) 
							break
						endif
						sprintf filename, "%03.0f_%04.0f.tsv", i, str2num(StringFromList(j, cycles))
						GetFileFolderInfo/Z/Q/P=CurrentRawPath filename
						if (V_Flag!=0)
							abort "Scan seems to be incomplete. File " + filename + " not found."
						endif
						LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentRawPath filename
						if (j==0)
							duplicate/o temp0, w_raw
						else
							w_raw += temp0
						endif
					endfor
					w_raw /= j
				endif
				wave w_conv = $ConvertImage(w_raw, passEnergy,kineticEnergy, LensMode, binning)
				w[][][i] = w_conv[p][q]
			endfor
			utils_progressDlg(message="Delay", done=1, numDone=0, numTotal=scansteps, title="Loading Scan...")
			Note/K w, ScanInfo + "\r" + settings
			variable TimeZero = NumberByKey("TimeZero", ScanInfo, "=", "\r")
			// deal with delay wave
			if (NumType(TimeZero)==0)
				w_scanvec -= TimeZero		// Time Zero
				w_scanvec *=2/3e11*1e15	// in fs
			endif
			if (utils_isEquidistant(w_scanvec))
				// Set time wavescaling if equidistant
				setScale/I z, w_scanvec[0], w_scanvec[numpnts(w_scanvec)-1], "fs", w
			else
				// Store in wavenote
				utils_addWaveNoteEntry(w, "DelayWave", nameofwave(w_scanvec))
				utils_addWaveNoteEntry(w, "ZAxisWave", nameofwave(w_scanvec))
			endif
					
			break
			
		case "delay_chop":
			// chopped delay scan
			// create datafolder for compatibility
			repetitions = NumberByKey("Repetitions", ScanInfo, "=", "\r")
			if (numtype(repetitions) !=0) 
				repetitions = 0
			endif			
			sprintf datafolder, "root:Data:R%03.0f_N%d", scan, repetitions
			NewDataFolder/O/S $datafolder
			// load scanvector
			filename = "scanvector.txt"
			LoadWave/Q/J/D/W/O/K=0/N=temp/P=CurrenRunPath filename
			//sprintf wname, "Data%03.0f_t", Scan
			// compatibility...
			wname="Data_t"
			wave temp0
			duplicate/o temp0, $wname
			wave w_scanvec = $wname
			scansteps = numpnts(w_scanvec)
			// load first image
			filename = "000_l.tsv"
			LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
			wave temp0
			wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
			//sprintf wname, "Data%03.0f", Scan
			// compatibility...
			wname="Data_l"
			make/o/n=(dimsize(w_conv, 0), dimsize(w_conv,1), scansteps) $wname
			wave w_l = $wname
			wname="Data_nl"
			make/o/n=(dimsize(w_conv, 0), dimsize(w_conv,1), scansteps) $wname
			wave w_nl = $wname
			wname="Data_diff"
			make/o/n=(dimsize(w_conv, 0), dimsize(w_conv,1), scansteps) $wname
			wave w_diff = $wname
			setscale/P x, dimoffset(w_conv, 0), dimdelta(w_conv, 0), waveunits(w_conv, 0), w_l, w_nl, w_diff 
			setscale/P y, dimoffset(w_conv, 1), dimdelta(w_conv, 1), waveunits(w_conv, 1), w_l, w_nl, w_diff 
			for (i=0; i<scansteps; i+=1)
				// with laser
				sprintf filename, "%03.0f_l.tsv", i
				LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
				wave temp0
				wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
				w_l[][][i] = w_conv[p][q]
				// without laser
				sprintf filename, "%03.0f_nl.tsv", i
				LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
				wave temp0
				wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
				w_nl[][][i] = w_conv[p][q]
			endfor
			w_diff[][] = w_l[p][q] - w_nl[p][q]
			Note/K w_l, ScanInfo
			Note/K w_nl, ScanInfo
			Note/K w_diff, ScanInfo
			TimeZero = NumberByKey("TimeZero", ScanInfo, "=", "\r")
			// deal with delay wave
			if (NumType(TimeZero)==0)
				w_scanvec -= TimeZero		// Time Zero
				w_scanvec *=2/3e11*1e15	// in fs
				utils_addWaveNoteEntry(w_l, "TimeZero", num2str(TimeZero))
				utils_addWaveNoteEntry(w_nl, "TimeZero", num2str(TimeZero))
				utils_addWaveNoteEntry(w_diff, "TimeZero", num2str(TimeZero))
			endif
			if (utils_isEquidistant(w_scanvec))
				// Set time wavescaling if equidistant
				setScale/I z, w_scanvec[0], w_scanvec[numpnts(w_scanvec)-1], "fs", w_l, w_nl, w_diff
			else
				// Store in wavenote
				utils_addWaveNoteEntry(w_l, "DelayWave", nameofwave(w_scanvec))
				utils_addWaveNoteEntry(w_l, "ZAxisWave", nameofwave(w_scanvec))
				utils_addWaveNoteEntry(w_nl, "DelayWave", nameofwave(w_scanvec))
				utils_addWaveNoteEntry(w_nl, "ZAxisWave", nameofwave(w_scanvec))
				utils_addWaveNoteEntry(w_diff, "DelayWave", nameofwave(w_scanvec))
				utils_addWaveNoteEntry(w_diff, "ZAxisWave", nameofwave(w_scanvec))
			endif
					
			wname="Data"
			duplicate/o w_diff, $wname
			wave w = $wname
			make/o/n=(dimsize(w, 0), dimsize(w,1)) w_temp2=0
			for (i=0; i<dimsize(w, 2); i+=1)
				w_temp2[][] += w_nl[p][q][i]/dimsize(w,2)
			endfor
			w[][][] = w_temp2[p][q] + w_diff[p][q][r]
					
			break
			
		case "single":
			// single image
			// create datafolder	
			//sprintf datafolder, "root:Data:R%03.0f", scan
			//NewDataFolder/O/S $datafolder
			SetDataFolder root:Data
			filename = "000.tsv"
			LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
			wave temp0
			wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
			sprintf wname, "Data%03.0f", Scan
			duplicate/o w_conv, $wname
			wave w = $wname
			Note/K w, ScanInfo
			if (keepRawData)
				NewDataFolder/O root:Data:RAW
				sprintf wname, ":RAW:RAWData%03.0f", Scan
				duplicate/o temp0, $wname
			endif
			break
			
		case "manipulator":
			// manipulator scan
			SetDataFolder root:Data
			// create datafolder for compatibility
			//string datafolder
			//variable repetitions = NumberByKey("Repetitions", ScanInfo, "=", "\n")
			//if (numtype(repetitions) !=0) 
			//	repetitions = 0
			//endif			
			//sprintf datafolder, "root:Data:R%03.0f_N%d", scan, repetitions
			//NewDataFolder/O/S $datafolder
			// load scanvector
			filename = "scanvector.txt"
			GetFileFolderInfo/Z/Q/P=CurrenRunPath filename
			if (V_Flag!=0)
				abort "scanvector.txt not found!"
			endif
			LoadWave/Q/J/D/W/M/O/K=0/N=temp/P=CurrenRunPath filename
			//sprintf wname, "Data%03.0f_t", Scan
			// compatibility...
			wname="Data_z"
			wave temp0
			variable index = find_moving_angle(temp0)
			make/o/n=(dimsize(temp0, 0)) $wname
			wave w_scanvec = $wname
			w_scanvec = temp0[p][index]
			scansteps = numpnts(w_scanvec)
			// load first image
			filename = "000.tsv"
			GetFileFolderInfo/Z/Q/P=CurrentAveragesPath filename
			if (V_Flag!=0)
				abort "Scan seems to be incomplete. File " + filename + " not found."
			endif
			LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
			wave temp0
			wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
			//sprintf wname, "Data%03.0f", Scan
			// compatibility...
			sprintf wname, "Data%03.0f", Scan
			make/o/n=(dimsize(w_conv, 0), scansteps, dimsize(w_conv,1)) $wname=0
			wave w = $wname
			setscale/P x, dimoffset(w_conv, 0), dimdelta(w_conv, 0), waveunits(w_conv, 0), w 
			setscale/P z, dimoffset(w_conv, 1), dimdelta(w_conv, 1), waveunits(w_conv, 1), w 
			utils_progressDlg(message="Delay", done=0, numDone=0, numTotal=scansteps, title="Loading Scan...")			
			for (i=0; i<scansteps; i+=1)
				if (utils_progressDlg(message="Scan Point " + num2str(i), done=0, numDone=i, numTotal=scansteps, title="Loading Scan...")) 
					break
				endif
				if (cmpstr(cycles, "") == 0)
					// load averages, as no Cycle selection has been done
					sprintf filename, "%03.0f.tsv", i
					GetFileFolderInfo/Q/Z/P=CurrentAveragesPath filename
					if (V_Flag!=0)
						print "File " + filename + " not found. Skipping Rest"
						i=scansteps
						continue
					endif
					LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
					wave temp0
					duplicate/o temp0, w_raw
				else
					// load RAW data, and select requested cycles
					for (j=0; j<itemsInList(cycles); j+=1)
						if (utils_progressDlg(message="Cycle " + num2str(str2num(StringFromList(j, cycles))), done=0, numDone=j, numTotal=itemsInList(cycles), title="Loading Scan...", level=1)) 
							break
						endif
						sprintf filename, "%03.0f_%04.0f.tsv", i, str2num(StringFromList(j, cycles))
						GetFileFolderInfo/Z/Q/P=CurrentRawPath filename
						if (V_Flag!=0)
							abort "Scan seems to be incomplete. File " + filename + " not found."
						endif
						LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentRawPath filename
						if (j==0)
							duplicate/o temp0, w_raw
						else
							w_raw += temp0
						endif
					endfor
					w_raw /= j
				endif
				wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
				w[][i][] = w_conv[p][r]
			endfor
			utils_progressDlg(message="Scan Point", done=1, numDone=0, numTotal=scansteps, title="Loading Scan...")
			Note/K w, ScanInfo
			if (utils_isEquidistant(w_scanvec, tolerance=0.5))
				// Set time wavescaling if equidistant
				setScale/I y, w_scanvec[0], w_scanvec[numpnts(w_scanvec)-1], w
			else
				// Store in wavenote
				utils_addWaveNoteEntry(w, "YAxisWave", nameofwave(w_scanvec))
			endif		
			// add all wave note entries
			filename = "scanvector.txt"
			string chunksecondangle
			switch (index)
				case 3: 
					chunksecondangle = "theta"
					break
				case 4: 
					chunksecondangle = "phi"
					break
				case 5: 
					chunksecondangle = "omega"
					break
				default:
					chunksecondangle = ""
					break
			endswitch
			LoadWave/Q/J/D/W/M/O/K=0/N=temp/P=CurrenRunPath filename
			utils_addWaveNoteEntry(w, "ManipulatorType", "flip")
			utils_addWaveNoteEntry(w, "InitialThetaManipulator", num2str(temp0[0][3]))
			utils_addWaveNoteEntry(w, "FinalThetaManipulator", num2str(temp0[dimsize(temp0,0)][3]))
			utils_addWaveNoteEntry(w, "OffsetThetaManipulator", num2str(-10))
			utils_addWaveNoteEntry(w, "InitialPhiManipulator", num2str(temp0[0][4]))
			utils_addWaveNoteEntry(w, "FinalPhiManipulator", num2str(temp0[dimsize(temp0,0)][4]))
			utils_addWaveNoteEntry(w, "OffsetPhiManipulator", num2str(0))
			utils_addWaveNoteEntry(w, "InitialOmegaManipulator", num2str(temp0[0][5]))
			utils_addWaveNoteEntry(w, "FinalOmegaManipulator", num2str(temp0[dimsize(temp0,0)][5]))
			utils_addWaveNoteEntry(w, "OffsetOmegaManipulator", num2str(0))
			utils_addWaveNoteEntry(w, "InitialAlphaAnalyzer", num2str(0))
			utils_addWaveNoteEntry(w, "FinalAlphaAnalyzer", num2str(0))
			utils_addWaveNoteEntry(w, "AngleSignConventions", num2str(8))
			utils_addWaveNoteEntry(w, "ChunkSecondAngle", chunksecondangle)
			utils_addWaveNoteEntry(w, "PhotonEnergy", num2str(21.8))
			utils_addWaveNoteEntry(w, "EnergyScale", "kinetic")
			utils_addWaveNoteEntry(w, "FermiLevel", num2str(21.8))
			utils_addWaveNoteEntry(w, "WorkFunction", num2str(4.558))
			utils_addWaveNoteEntry(w, "ScientaOrientation", num2str(270))
			break
			
			
			
					//Kenichi-Gian: Automatized FS **14/09/2017**
		case "yRotation":
			// angle scan
			// create datafolder for compatibility
			if (numtype(repetitions) !=0) 
				repetitions = 0
			endif			
			sprintf datafolder, "root:Data:R%03.0f_N%d", scan, repetitions
			NewDataFolder/O/S $datafolder
			// load scanvector
			filename = "scanvector.txt"
			GetFileFolderInfo/Z/Q/P=CurrenRunPath filename
			if (V_Flag!=0)
				abort "scanvector.txt not found!"
			endif
			LoadWave/Q/J/D/W/O/K=0/N=temp/P=CurrenRunPath filename
			//sprintf wname, "Data%03.0f_t", Scan
			// compatibility...
			wname="Data_t"
			wave temp0
			duplicate/o temp0, $wname
			wave w_scanvec = $wname
			scansteps = numpnts(w_scanvec)
			// load first image
			filename = "000.tsv"
			GetFileFolderInfo/Z/Q/P=CurrentAveragesPath filename
			if (V_Flag!=0)
				abort "Scan seems to be incomplete. File " + filename + " not found."
			endif
			LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
			wave temp0
			wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
			//sprintf wname, "Data%03.0f", Scan
			// compatibility...
			wname="Data"
			make/o/n=(dimsize(w_conv, 0), dimsize(w_conv,1), scansteps) $wname
			wave w = $wname
			setscale/P x, dimoffset(w_conv, 0), dimdelta(w_conv, 0), waveunits(w_conv, 0), w 
			setscale/P y, dimoffset(w_conv, 1), dimdelta(w_conv, 1), waveunits(w_conv, 0), w 
			for (i=0; i<scansteps; i+=1)
				sprintf filename, "%03.0f.tsv", i
				GetFileFolderInfo/Z/Q/P=CurrentAveragesPath filename
				if (V_Flag!=0)
					abort "Scan seems to be incomplete. File " + filename + " not found."
				endif
				LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
				wave temp0
				wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
				w[][][i] = w_conv[p][q]
			endfor
			Note/K w, ScanInfo
			// deal with delay wave
			
			setScale/I z, w_scanvec[0], w_scanvec[numpnts(w_scanvec)-1], "deg", w
			
					
			break
			
		default:
			// everything else
			abort "not supported"
			break		
	endswitch
	
	killwaves/Z temp0, w_raw, w_conv
	setDataFolder $DF
	return getWavesDataFolder(w, 2)
end


function/S CheckScan(Path, Scan, Delay, [p1, p2, q1, q2, q3, q4])
	string Path
	Variable Scan, Delay
	variable p1, p2, q1, q2, q3, q4
	
	Variable passEnergy, kineticEnergy, LensMode, binning
	
	NVAR keepRawData = root:Data:tmpData:Analyser:gv_keepRawData
	
	if (cmpstr(Path, "")!=0)
		// Path given, overwrite old
		ButtonChoosePath("", Path=Path)
	else
		// Path not given, look at Variable and set it to it. This will open a Prompt, if it is not set
		SVAR trARPES_load_path = root:Data:tmpData:Analyser:gs_trARPES_load_path
		ButtonChoosePath("", Path=trARPES_load_path)
	endif
	// check the path
	PathInfo/S All_Runs_Path
	if (V_flag==0)
		// path does not exist, Run fuction to get it
		ButtonChoosePath("")
	endif
	string foldername
	sprintf foldername, "%04.0f", Scan
	GetFileFolderInfo/Z/Q/P=All_Runs_Path foldername
	if (V_Flag!=0)
		abort "Scan not found!"
	endif
	NewPath/O/Z/Q CurrenRunPath, S_Path
	NewPath/O/Z/Q CurrentRawPath, S_Path + "RAW"
	// get info.txt
	string filename = "info.txt"
	GetFileFolderInfo/Z/Q/P=CurrenRunPath filename
	if (V_Flag!=0)
		abort "info.txt not found."
	endif
	string ScanInfo = parseInfoTxt(S_Path)	
	// get lens mode
	string s_lensmode = StringByKey("LensMode", ScanInfo, "=", "\r")
	if (cmpstr(s_lensmode, "")==0)
		s_lensmode = StringByKey("Mode", ScanInfo, "=", "\r")
	endif
	strswitch(s_lensmode)	
		case "LowAngularDispersion":
			LensMode=0
			break
		case "WideAngleMode":
			LensMode=3
			break
		default:
			abort "Lens mode " + StringByKey("LensMode", ScanInfo, "=", "\r") + " not implemented"
	endswitch
	binning = 2^NumberByKey("Binning", ScanInfo, "=", "\r")
	nvar old_binning=root:Data:tmpData:Analyser:binning
	if (binning!=old_binning)
		// set angular center pixel
		NVAR Ang_Centre_px = root:Data:tmpData:Analyser:Ang_Centre_px
		Ang_Centre_px = 504/binning
		old_binning = binning
	endif
	DoWindow Gr_ImportData_Setting // Check if Window exists
	if (V_flag == 1)
		PopupMenu Popup_LensMode win=Gr_ImportData_Setting, mode=LensMode+1
		popupMenu_Binning("", 0, "", set_binning=binning)
	endif
	//Pass energy
	passEnergy = NumberByKey("PassEnergy", ScanInfo, "=", "\r")
	kineticEnergy = NumberByKey("KineticEnergy", ScanInfo, "=", "\r")
	string wname
	variable scansteps
	variable i
	string DF = getDataFolder(1)
	// What kind of scan?
	strswitch(StringByKey("ScanType", ScanInfo, "=", "\r"))
		case "delay":
			// delay scan
			// create datafolder for compatibility
			string datafolder
			variable repetitions = NumberByKey("Repetitions", ScanInfo, "=", "\r")
			if (numtype(repetitions) !=0) 
				repetitions = 0
			endif			
			sprintf datafolder, "root:Data:R%03.0f_N%d", scan, repetitions
			NewDataFolder/O/S $datafolder
			// load scanvector
			filename = "scanvector.txt"
			GetFileFolderInfo/Z/Q/P=CurrenRunPath filename
			if (V_Flag!=0)
				abort "scanvector.txt not found!"
			endif
			LoadWave/Q/J/D/W/O/K=0/N=temp/P=CurrenRunPath filename
			//sprintf wname, "Data%03.0f_t", Scan
			// compatibility...
			wname="Data_t"
			wave temp0
			//duplicate/o temp0, $wname
			wave w_scanvec = temp0
			scansteps = numpnts(w_scanvec)
			if (Delay > scansteps)
				abort "Delay not valid!"
			endif
			// load first image
			sprintf filename, "%03.0f_0000.tsv", delay
			GetFileFolderInfo/Z/Q/P=CurrentRawPath filename
			if (V_Flag!=0)
				abort "Scan seems to be incomplete. File " + filename + " not found."
			endif
			LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentRawPath filename
			wave temp0
			wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
			//sprintf wname, "Data%03.0f", Scan
			// find last repetition
			variable num_repetition=0
			do
				sprintf filename, "%03.0f_%04.0f.tsv", delay, num_repetition
				num_repetition +=1
				GetFileFolderInfo/Z/Q/P=CurrentRawPath filename
			while (V_Flag==0)
			num_repetition -=1
			// compatibility...
			sprintf wname, "Data_d%03.0f_check", delay
			make/o/n=(dimsize(w_conv, 0), dimsize(w_conv,1), num_repetition) $wname
			wave w = $wname
			setscale/P x, dimoffset(w_conv, 0), dimdelta(w_conv, 0), "", w 
			setscale/P y, dimoffset(w_conv, 1), dimdelta(w_conv, 1), "", w 
			for (i=0; i<num_repetition; i+=1)
				sprintf filename, "%03.0f_%04.0f.tsv", delay, i
				GetFileFolderInfo/Z/Q/P=CurrentRawPath filename
				if (V_Flag!=0)
					abort "Scan seems to be incomplete. File " + filename + " not found."
				endif
				LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentRawPath filename
				wave temp0
				wave w_conv = $ConvertImage(temp0, passEnergy,kineticEnergy, LensMode, binning)
				w[][][i] = w_conv[p][q]
			endfor

			break
	endswitch
	
	killwaves/Z temp0, w_conv
	
	// get cuts and display
	if (!paramIsDefault(p1) && !paramIsDefault(p2) && !paramIsDefault(q1) & !paramIsDefault(q2) )
		ImageStats/BEAM/RECT={p1, p2, q1, q2}/M=1 w
		Wave W_ISBeamAvg
		wname = nameofwave(w) + "_cut1"
		duplicate/o W_ISBeamAvg $wname
		wave w_cut1 = $wname
		string gname = "w_scan" + num2str(scan) + "_check"
		doWindow/K $gname
		display as gname
		DoWindow/C $gname
		appendToGraph w_cut1
		if (!paramIsDefault(q3) & !paramIsDefault(q4))
			// cut 2
			ImageStats/BEAM/RECT={p1, p2, q3, q4}/M=1 w
			Wave W_ISBeamAvg
			wname = nameofwave(w) + "_cut2"
			duplicate/o W_ISBeamAvg $wname
			wave w_cut2 = $wname
			appendToGraph w_cut2
			ModifyGraph log(left)=1
			wname = nameofwave(w) + "_cutRatio"
			duplicate/o w_cut1, $wname
			wave w_cutratio= $wname
			w_cutratio = w_cut2/w_cut1
			appendToGraph/R w_cutratio
			ModifyGraph rgb($wname)=(0,0,0)
		endif
	endif
	
	setDataFolder $DF
	return getWavesDataFolder(w, 2)
end



function find_moving_angle(w)
	wave w
	
	variable i, index
	make/n=6/o w_dimensions=0
	for (i=0; i<dimsize(w,1); i+=1)
		if (w[0][i] != w[1][i])
			w_dimensions[i]=1
		endif
	endfor
	if (sum(w_dimensions, 3, 5) > 0)
		for (i=3; i<6; i+=1)
			if (w_dimensions[i] == 1)
				return i
			endif
		endfor
	else
		for (i=0; i<3; i+=1)
			if (w_dimensions[i] == 1)
				return i
			endif
		endfor
	endif
	return -1
end

Function/S parseInfoTxt(infoTxtPath)
	string infoTxtPath	
	LoadWave/O/J/Q/K=2/N=w_str infoTxtPath
	wave/T w_str0
	string ret=""
	variable i	
	string temp
	temp=w_str0[0]
	String SeperatorString
	// compatibility for either ":" or "=" seperator strings
	if (strsearch(temp, "=", 0) > 0)
		// contains "=", use this preferentially
		SeperatorString="="
	else
		//does not contain "=", so use ":"
		SeperatorString=":"
	endif
	for (i=0; i<numpnts(w_str0); i+=1)
		temp=w_str0[i] 
		sprintf ret, "%s\r%s=%s", ret, temp[0, strsearch(temp, SeperatorString, 0)-1],  temp[strsearch(temp, SeperatorString, 0)+1, strlen(temp)-1]
	endfor
	killwaves/Z w_str0
	return ret
end	
	
	
function/S loadOldData(Path, RunNo)
	string Path
	variable RunNo

	String Str, FolderList, FolderName, FileList, DataFolder, Str_LensMode
	Variable i, j, N, refNum, TimeZero
	
	SetDataFolder root:Data:tmpData:'Analyser':
	NVAR LensMode, Ek, Ep, binning


	sprintf Str, "%003.0f", RunNo
	DataFolder = "root:Data:'R"+Str		//	for Data folder name in experiment

	if (cmpstr(Path, "")!=0)
		// Path given, overwrite old
		ButtonChoosePath("", Path=Path)
	else
		// Path not given, look at Variable and set it to it. This will open a Prompt, if it is not set
		SVAR trARPES_load_path = root:Data:tmpData:Analyser:gs_trARPES_load_path
		ButtonChoosePath("", Path=trARPES_load_path)
	endif
	// check the path
	PathInfo/S All_Runs_Path
	if (V_flag==0)
		// path does not exist, Run fuction to create it
		ButtonChoosePath("")
	else
		// TODO: Check if the scan is there		
		// check for subdirectories now
		FolderList=IndexedDir(All_Runs_Path, -1, 0)
		if (cmpstr(FolderList, "")==0)
			ButtonChoosePath("")
		endif
	endif

	FolderList=IndexedDir(All_Runs_Path, -1, 0)

	FolderName=StringFromList(0, ListMatch(IndexedDir(All_Runs_Path, -1, 0), Str+" *"))

	if (StringMatch(FolderName, "")==1)
		// No Subfolders found, abort
		Abort
	endif
	PathInfo/S All_Runs_Path
	NewPath/O/C/Q/Z Run_Path, ParseFilePath(5, S_path, "\\", 0, 0)+FolderName+"\\"

	NVAR setN = root:Data:tmpData:Analyser:setN
	N=setN
	if (numType(N) == 2)
		N=0
		FolderList=IndexedDir(Run_Path, -1, 0)
		i=0
		do	// Find max N
			j=Str2Num(ReplaceString("N=", StringFromList(i, FolderList),""))
			if ( NumType(j) != 0 )
				break
			else
				if ( j>N )
					N=j
				endif
			endif
			i+=1
		while(1)
	endif
	PathInfo/S Run_Path
	NewPath/O/C/Q/Z RawData_Path, ParseFilePath(5, S_path, "\\", 0, 0)+" N="+Num2Str(N)+"\\"  //Claude: modified 03.07.14: space added before "N="
	if (V_Flag!=0)	
		Abort
	endif
	
	FileList=SortList(Indexedfile(RawData_Path, -1, ".tsv"), ";", 2)

//	Data path on Drive
	PathInfo/S RawData_Path
	Str = StrSubstitute(":Raw Data:",S_path ,":data:")
	NewPath/O/C/Q/Z Data_path, Str	// Create the folder for Data on the drive
	if (V_Flag!=0)		// Create folder by folder on the drive
		i=1
		do
			NewPath/O/C/Q/Z Data_path, ParseFilePath(1, Str, ":", 0, i)
			i+=1
			NewPath/O/C/Q/Z Data_path, Str
		while (V_Flag!=0)
	endif

//	Data Folder in the experiment
	DataFolder += "_N"+Num2Str(N)+"'"
	if (DataFolderExists(DataFolder)==1)
		KillDataFolder/Z DataFolder
		NewDataFolder/O $DataFolder
	else
		Str = "root"
		i=1
		do
			Str += ":" + StringFromList(i, DataFolder, ":")
			if (DataFolderExists(Str)!=1)
				NewDataFolder/O $Str
			endif
			i+=1
		while (DataFolderExists(DataFolder)!=1)
	endif
	
	Wave Angular_Correction=root:Data:tmpData:Analyser:Angular_Correction
	open/Z/R/P=Run_path refNum as "Info.tsv"
	if (StringMatch(S_fileName, "") != 1)
		Str = PadString(Str, 5000, 0)
		FBinRead refNum, Str
		close refNum
		Ek=NumberByKey("KE", Str)
		Ep=NumberByKey("PE", Str)
		//Binning=NumberByKey("Binning", Str)
		TimeZero=NumberByKey("Time Zero", Str)
		Str_LensMode=StringByKey("LensMode", Str)
		strswitch(Str_LensMode)	// string switch
			case "LowAngularDispersion":
				LensMode=0
				break
			case "WideAngleMode":
				LensMode=3
				break
		endswitch
		DoWindow Gr_ImportData_Setting // Check if Window exists
		if (V_flag == 1)
			PopupMenu Popup_LensMode win=Gr_ImportData_Setting, mode=LensMode+1
		endif
	endif
	if ( (NumberByKey("LensMode",Note(Angular_Correction))!=LensMode) || (NumberByKey("Ek",Note(Angular_Correction))!=Ek) || (NumberByKey("PE",Note(Angular_Correction))!=Ep) || (NumberByKey("binning",Note(Angular_Correction))!=binning) )
		Calculate_Da_values()
		Calculate_Polynomial_Coef_Da()
		Calculate_MatrixCorrection()
		// disable crop if it was selected
		print "New Angular Correction Matrix"
		NVAR Crop = root:Data:tmpData:Analyser:Crop
		if (Crop==1)
			Crop = 0
			print "Crop has been disabled, revisit Settings window to enable!"
		endif
	endif

	SetDataFolder $DataFolder

	Import_LabViewData(FileList)

	if (NumType(TimeZero)==0)
		Wave Data, Data_t
		Data_t -= TimeZero		// Time Zero
		Data_t *=2/3e11*1e15	// in fs
		Str=ReplaceNumberByKey("Time Zero", Note(Data), TimeZero)
		Note/K Data, Str
		// Set time wavescaling if equidistant
		if (utils_isEquidistant(Data_t))
			setScale/I z, Data_t[0], Data_t[numpnts(Data_t)-1], Data
		endif
	endif


	SetDataFolder $DataFolder
	
	return getWavesDataFolder(Data,2)
end

function ButtonChoosePath(ctrlName, [Path]) : ButtonControl
	String ctrlName, Path
	if (paramIsDefault(Path) || cmpstr(Path, "")==0)
		// no Path given, ask user to pick
		NewPath/O/Q/M="Choose Experimental Folder" All_Runs_Path
		if (V_flag!=0)
			// something went wrong)
			abort
		endif
	else
		// Path given, set it to it
		NewPath/Z/O/Q All_Runs_Path, Path
		if (V_flag !=0)
			// folder not valid, ask user to choose
			NewPath/O/Q/M="Choose Experimental Folder" All_Runs_Path
			if (V_flag!=0)
				// something went wrong)
				abort
			endif
		endif
	endif
	PathInfo All_Runs_Path
	if (V_flag ==0)
		//folder not found, let user Choose
		NewPath/O/Q/M="Choose Experimental Folder" All_Runs_Path
		if (V_flag!=0)
			// something went wrong)
			abort
		endif
		PathInfo All_Runs_Path
		if (V_flag==0)
			// now it is good...
			abort "An error occured"
		endif
	endif
	// Put Path in Box
	SVAR trARPES_load_path = root:Data:tmpData:Analyser:gs_trARPES_load_path
	trARPES_load_path = S_path
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function Import_LabViewData(Str_Filename_List)
String Str_Filename_List

	NVAR EkinLow = root:Data:tmpData:Analyser:EkinLow
	NVAR EkinHigh = root:Data:tmpData:Analyser:EkinHigh
	NVAR AzimuthLow = root:Data:tmpData:Analyser:AzimuthLow
	NVAR AzimuthHigh = root:Data:tmpData:Analyser:AzimuthHigh
	NVAR Edge_Correction=root:Data:tmpData:Analyser:Edge_Correction
	NVAR Crop=root:Data:tmpData:Analyser:Crop
	NVAR Check_2D_Smooth=root:Data:tmpData:Analyser:Check_2D_Smooth

	Wave Correction_Matrix=root:Data:tmpData:Analyser:Correction_Matrix
	
	String Path, DataFolder, w_Name, FileList, Str
	Variable numFilesSelected, i
	
	String Str_PathFilename_List, Str_PathFilename, Str_Filename

	if(StringMatch(Str_Filename_List, "")==1)
		Abort
	endif
	Str_Filename_List = StrSubstitute("info.tsv;",Str_Filename_List ,"")
	PathInfo RawData_Path
	Path=S_path

	numFilesSelected = ItemsInList(Str_Filename_List, ";")

	Str_PathFilename = Path+StringFromList(0, Str_Filename_List, ";")
	LoadWave/Q/J/M/D/K=1/N=LoadW Str_PathFilename
	Wave LoadW0

	if ( Crop==1 )
		NVAR Delete_nE=root:Data:tmpData:Analyser:delete_nE
		NVAR Delete_nTheta=root:Data:tmpData:Analyser:Delete_nTheta
		NVAR NewDim_E=root:Data:tmpData:Analyser:NewDim_E
		NVAR NewDim_Theta=root:Data:tmpData:Analyser:NewDim_Theta
		NVAR NewOffset_E=root:Data:tmpData:Analyser:NewOffset_E
		NVAR NewOffset_Theta=root:Data:tmpData:Analyser:NewOffset_Theta
		NVAR Delta_E=root:Data:tmpData:Analyser:Delta_E
		NVAR Delta_Theta=root:Data:tmpData:Analyser:Delta_Theta

		Make/O/N=(NewDim_Theta, NewDim_E, numFilesSelected) Data		//	Data will be transposed in (Theta, E)
		//Make/O/N=(NewDim_E+1) Data_E
		//Make/O/N=(NewDim_Theta+1) Data_k
		// Internal wave scaling
		Setscale/P y, NewOffset_E, Delta_E, Data
		Setscale/P x, NewOffset_Theta, Delta_Theta, Data
		
		Make/O/N=(numFilesSelected) Data_t

		//Data_E[]=NewOffset_E+p*Delta_E
		//Data_k[]=NewOffset_Theta+p*Delta_Theta
		
	else

		if ( DimSize(LoadW0,0) < DimSize(LoadW0, 1) )		// TransposeMatrix in (E, Theta) the orientation of the Data matrix
			MatrixTranspose LoadW0
		endif
		Make/O/N=(DimSize(LoadW0, 1), DimSize(LoadW0, 0), numFilesSelected) Data		//	Data will be transposed in (Theta, E)
		//Make/O/N=(DimSize(LoadW0, 0)+1) Data_E
		//Make/O/N=(DimSize(LoadW0, 1)+1) Data_k
		// Internal wave scaling
		Duplicate/O LoadW0 LoadW1	// Angular correction
		PhysicalUnits_Data(LoadW1, LoadW0)
		Setscale/P y, dimOffset(LoadW0,0), dimDelta(LoadW0,0), Data
		Setscale/P x, dimOffset(LoadW0,1), dimDelta(LoadW0,1), Data
		Make/O/N=(numFilesSelected) Data_t

		//Data_E[]=EkinLow+p*(EkinHigh-EkinLow)/(DimSize(LoadW0, 0)-1)
		//Data_k[]=AzimuthLow+p*(AzimuthHigh-AzimuthLow)/(DimSize(LoadW0, 1)-1)
	endif
	
	if (Edge_Correction)
		String fldrSav0= GetDataFolder(1)
		Scaling_Correction()
		SetDataFolder fldrSav0
	endif
	
	for(i=0; i<numFilesSelected; i+=1)
		Str_Filename=StringFromList(i, Str_Filename_List, ";")
		Str_PathFilename = path+Str_Filename
		//  Load ,tsv Data
		LoadWave/Q/J/M/D/K=1/N=LoadW Str_PathFilename
		if ( DimSize(LoadW0,0) < DimSize(LoadW0, 1) )		// TransposeMatrix in (E, Theta)
			MatrixTranspose LoadW0
		endif
		
		Data_t[i]=Str2Num(ParseFilePath(3, Str_Filename, ":", 0, 0))
		
		
		///////////////////MICHELE EDIT HERE
		NVAR filter_Image=root:Data:tmpData:Analyser:filter_Image
		if (filter_Image)
		NVAR binning = root:Data:tmpData:Analyser:binning
		NVAR fx=root:Data:tmpData:Analyser:filter_fx
		NVAR fy=root:Data:tmpData:Analyser:filter_fy
		NVAR wx=root:Data:tmpData:Analyser:filter_wx
		NVAR wy=root:Data:tmpData:Analyser:filter_wy
		NVAR A=root:Data:tmpData:Analyser:filter_A
		filt2dfancy(LoadW0, fx*binning, fy*binning, wx*binning, wy*binning, A)	 
		endif	
		///////////////////////////////////////////////////////		
		
		Duplicate/O LoadW0 LoadW1	// Angular correction
		PhysicalUnits_Data(LoadW1, LoadW0)
				
		if ( Check_2D_Smooth )
			SmoothingImage(LoadW0)
		endif
		
		if ( Crop )
			DeletePoints/M=0 0,Delete_nE, LoadW0
			DeletePoints/M=1 0,Delete_nTheta, LoadW0
		endif

		
		MatrixTranspose LoadW0							// TransposeMatrix in (Theta, E)	
		
		
		
		MultiThread  Data[][][i]=LoadW0[p][q]

	endfor
	
	// LR I think we do not need this.
//	f (DimSize(Data, 2)<4)
//		Redimension/N=(-1,-1,5) Data
//	endif


	
	//  Save waves
	Write_DataNote(Data)
	//Save/O/P=Data_path Data Data_E Data_k Data_t
	Save/O/P=Data_path Data Data_t
	KillWaves/Z LoadW0,LoadW1

end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function Read_Info_File()

	Variable refNum
	String Str="", Str_LensMode=""

	SetDataFolder root:Data:tmpData:'Analyser':
	NVAR LensMode, Ek, Ep, binning

	nvar dataFormat = gv_DataFormat
	switch (dataFormat)
		case 1: // old data
			open/Z/R/P=Run_path refNum as "Info.tsv"
			if (StringMatch(S_fileName, "") != 1)
				Str = PadString(Str, 5000, 0)
				FBinRead refNum, Str
				close refNum
				Ek=NumberByKey("KE", Str)
				Ep=NumberByKey("PE", Str)
				Str_LensMode=StringByKey("LensMode", Str)
			endif
			break
		case 2: // New data
			string filename = "info.txt"
			GetFileFolderInfo/Z/Q/P=CurrenRunPath filename
			if (V_Flag!=0)
				abort
			endif
			string ScanInfo = parseInfoTxt(S_Path)	
			Ek = NumberByKey("KineticEnergy", ScanInfo, "=", "\r")
			EP = NumberByKey("PassEnergy", ScanInfo, "=", "\r")
			Binning = 2^NumberByKey("Binning", ScanInfo, "=", "\r")
			Str_LensMode = StringByKey("Mode", ScanInfo, "=", "\r")
			break
		default:
			abort "Data Format not recognized!"
	endswitch
	strswitch(Str_LensMode)	// string switch
		case "LowAngularDispersion":
			LensMode=0
			break
		case "WideAngleMode":
			LensMode=3
			break
	endswitch
	DoWindow Gr_ImportData_Setting // Check if Window exists
	if (V_flag == 1)
		PopupMenu Popup_LensMode win=Gr_ImportData_Setting, mode=LensMode+1
		popupMenu_Binning("",0,"",set_binning=Binning)
	endif
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function Write_DataNote(Data)
Wave Data

	NVAR EkinLow = root:Data:tmpData:Analyser:EkinLow
	NVAR EkinHigh = root:Data:tmpData:Analyser:EkinHigh
	NVAR AzimuthLow = root:Data:tmpData:Analyser:AzimuthLow
	NVAR AzimuthHigh = root:Data:tmpData:Analyser:AzimuthHigh

	NVAR n_size=root:Data:tmpData:Analyser:n_size
	NVAR passes=root:Data:tmpData:Analyser:passes

	NVAR Delete_nE=root:Data:tmpData:Analyser:delete_nE
	NVAR Delete_nTheta=root:Data:tmpData:Analyser:Delete_nTheta
	NVAR NewDim_E=root:Data:tmpData:Analyser:NewDim_E
	NVAR NewDim_Theta=root:Data:tmpData:NewDim_Theta
	NVAR NewOffset_E=root:Data:tmpData:Analyser:NewOffset_E
	NVAR NewOffset_Theta=root:Data:tmpData:Analyser:NewOffset_Theta
	NVAR Delta_E=root:Data:tmpData:Analyser:Delta_E
	NVAR Delta_Theta=root:Data:tmpData:Analyser:Delta_Theta

	String DataNote, str_path, RawDataFolder, DataFolder, FolderName, Nb_frames, Cropping

	PathInfo RawData_path
	str_path=S_path

	RawDataFolder = str_path[strsearch(str_path, "Data Analysis:Raw Data:", 0)+14, strlen(str_path)-1]
	DataFolder = str_path[strsearch(str_path, "Data Analysis:Raw Data:", 0)+18, strlen(str_path)-1]
	FolderName=ParseFilePath(0, str_path, ":", 1, 0)
	Nb_frames=Num2Str(Str2Num(FolderName[strsearch(FolderName, "N=", 0)+2, strlen(FolderName)-1]))
	Cropping=Num2Str(Delete_nE)+"#"
	Cropping+=Num2Str(Delete_nTheta)+"#"
	Cropping+=Num2Str(NewDim_E)+"#"
	Cropping+=Num2Str(NewDim_Theta)+"#"
	Cropping+=Num2Str(NewOffset_E)+"#"
	Cropping+=Num2Str(NewOffset_Theta)+"#"
	Cropping+=Num2Str(Delta_E)+"#"
	Cropping+=Num2Str(Delta_Theta)
	
	//  Data Note
	DataNote=";DataType=Tr-ARPES\r"
	DataNote+="RawDataPath="+RawDataFolder+"\r"
	DataNote+="DataPath="+DataFolder+"\r"
	DataNote+="FolderName="+FolderName+"\r"
	DataNote+="~~Analyser~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r"
	DataNote+="NbFrames="+Nb_frames+"\r"
	NVAR EkinLow = root:Data:tmpData:Analyser:EkinLow
	NVAR EkinHigh = root:Data:tmpData:Analyser:EkinHigh
	NVAR AzimuthLow = root:Data:tmpData:Analyser:AzimuthLow
	NVAR AzimuthHigh = root:Data:tmpData:Analyser:AzimuthHigh

	DataNote+="Emin:"+Num2Str(EkinLow)+";Emax:"+Num2Str(EkinHigh)+";Thetamin:"+Num2Str(AzimuthLow)+";Theta Max:"+Num2Str(AzimuthHigh)+";\r"
	DataNote+="~~Image Processing~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r"
	DataNote+=";Smoothing:"+Num2Str(n_size)+"#"+Num2Str(passes)+";Cropping:"+Cropping+";\r"
	DataNote+=";Background Substraction:;\r"
	DataNote+="~~ARPES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r"
	DataNote+=";Theta manipulator:;Theta Zero:;Theta:;Lattice:;Ef:;\r"
	DataNote+=";Time Zero:;\r"
	
	DataNote+="DelayWave=Data_t\r"
	
	Note/K Data, DataNote
end

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// returns theStr with all instances of srcPat in theStr replaced with destPat
// note: Search is case sensitive
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
static Function/S StrSubstitute(srcPat,theStr,destPat)
	String srcPat,theStr,destPat
	
	Variable sstart=0,sstop,srcLen= strlen(srcPat), destLen= strlen(destPat)
	do
		sstop= strsearch(theStr, srcPat, sstart)
		if( sstop < 0 )
			break
		endif
		theStr[sstop,sstop+srcLen-1]= destPat
		sstart= sstop+destLen
	while(1)
	return theStr
End





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////// SETTINGS //////////////////////////////////////////////////////////////////////////////////////

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function ImportData_Settings(ctrlName) : ButtonControl
	String ctrlName

	SetDataFolder root:Data:tmpData:'Analyser':

	NVAR LensMode, Ek, Ep
	
	Variable refNum
	String Str="", Str_LensMode
	
	SetDataFolder root:Data:tmpData:

	Make/O/N=(2,2) RawData, PhysicalUnitsData, ARPESData
	Make/O/N=5 CropLine_x=NaN, CropLine=NaN
	//Make/O/N=(3) PhysicalUnitsData_E, PhysicalUnitsData_Ang
	//PhysicalUnitsData_E[]=p
	//PhysicalUnitsData_Ang[]=p

	DoWindow/F Gr_ImportData_Setting
	if (v_flag == 0)
		Execute "Gr_ImportData_Setting()"
		DoWindow/T Gr_ImportData_Setting, "Settings to import Tr-ARPES Data"
	endif
	LoadFirstImage("")

	Read_Info_File()
	
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function LoadFirstImage(ctrlName) : ButtonControl
	String ctrlName

	String Str, FolderList, FolderName
	Variable N, i, j
	
	SetDataFolder root:Data:tmpData:
	ControlInfo/W=Import_TrARPES_Data setRunNb
	variable RunNumber = V_value
	sprintf Str, "%003.0f", V_value

	NVAR dataFormat = root:Data:tmpData:Analyser:gv_DataFormat

	switch(dataFormat)
		case 1: //old Program
			FolderList=IndexedDir(All_Runs_Path, -1, 0)
			FolderName=StringFromList(0, ListMatch(FolderList, Str+" *")) 
			if (StringMatch(FolderName, "")==1)
				Abort
			endif
			PathInfo/S All_Runs_path
			NewPath/O/C/Q/Z Run_path, ParseFilePath(5, S_path, "\\", 0, 0)+FolderName+"\\"

			ControlInfo/W=Import_TrARPES_Data setN
			N=V_Value
			if (numType(N) == 2)
				N=0
				FolderList=IndexedDir(Run_path, -1, 0)
				i=0
				do	// Find max N
					j=Str2Num(ReplaceString("N=", StringFromList(i, FolderList),""))
					if ( NumType(j) != 0 )
						break
					else
						if ( j>N )
							N=j
						endif
					endif
					i+=1
				while(1)
			endif
			PathInfo/S Run_path
			NewPath/O/C/Q/Z RawData_path, ParseFilePath(5, S_path, "\\", 0, 0)+" N="+Num2Str(N)+"\\"   //Claude: modified 03.07.14: space added before "N="
			if (V_Flag!=0)	
				Abort
			endif

			Str=Indexedfile(RawData_path, 0, ".tsv")
	
			LoadWave/Q/J/M/D/K=1/P=RawData_path Str 
			S_waveNames = StringFromList(0, S_waveNames, ";")
			wave w =  $S_waveNames
			break
		case 2: //New Program
			loadNewData("", RunNumber)
			string filename = "000.tsv"
			LoadWave/Q/J/M/D/N=temp/K=0/P=CurrentAveragesPath filename
			wave w = temp0
			break
		default:
			abort "Data Format not recognized!"
	endswitch
	Duplicate/O w RawData
	Duplicate/O w PhysicalUnitsData
	Duplicate/O w ARPESData
	KillWaves/Z w
	PhysicalUnitsData=NaN
	ARPESData=NaN
	
	if ( DimSize(RawData,0) < DimSize(RawData,1) )
		MatrixTranspose RawData
		MatrixTranspose PhysicalUnitsData
		MatrixTranspose ARPESData
	endif
	
//	Make/O/N=(DimSize(RawData, 0)+1) PhysicalUnitsData_E
//	Make/O/N=(DimSize(RawData, 1)+1) PhysicalUnitsData_Ang
//	PhysicalUnitsData_E[]=p
//	PhysicalUnitsData_Ang[]=p
	Make/O/N=5 CropLine_x=NaN, CropLine=NaN
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function LoadExternalImage(ctrlName) : ButtonControl
	String ctrlName

	String Str, FolderList, FolderName
	Variable N, i, j
	
	SetDataFolder root:Data:tmpData:
	String cmd = "CreateBrowser prompt=\"select a wave and click 'ok'\""
	execute cmd
	SVAR S_BrowserList=S_BrowserList
	variable NumWaves = ItemsInList(S_BrowserList)
	if (NumWaves >= 1)
		// TODO: check wave dimensions
		wave w = $stringFromList(0, S_BrowserList)
		if (wavedims(w) != 2)
			return -1
		endif
	else
		return -1
	endif
	
	Duplicate/O w RawData
	Duplicate/O w PhysicalUnitsData
	Duplicate/O w ARPESData
	PhysicalUnitsData=NaN
	ARPESData=NaN
	
	if ( DimSize(RawData,0) < DimSize(RawData,1) )
		MatrixTranspose RawData
		MatrixTranspose PhysicalUnitsData
		MatrixTranspose ARPESData
	endif
	
	Make/O/N=5 CropLine_x=NaN, CropLine=NaN
end



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function Button_PhysicalUnits(ctrlName) : ButtonControl
	String ctrlName

	SetDataFolder root:Data:tmpData:
	Wave RawData, PhysicalUnitsData, ARPESData

	NVAR Binning=root:Data:tmpData:Analyser:Binning
	NVAR E_Offset_px=root:Data:tmpData:Analyser:E_Offset_px
	NVAR Ang_Centre_px=root:Data:tmpData:Analyser:Ang_Centre_px
	NVAR Ang_Offset_px=root:Data:tmpData:Analyser:Ang_Offset_px
	NVAR LensMode=root:Data:tmpData:Analyser:LensMode
	NVAR Edge_Correction=root:Data:tmpData:Analyser:Edge_Correction
	E_Offset_px=0
	Ang_Offset_px=Ang_Centre_px-1040/Binning/2
	ControlInfo/W=Gr_ImportData_Setting Popup_LensMode
	LensMode = v_Value-1
	
	Calculate_Da_values()
	Calculate_Polynomial_Coef_Da()
	Calculate_MatrixCorrection()
	PhysicalUnits_Data(RawData, PhysicalUnitsData)

	SetDataFolder root:Data:tmpData:
	Wave PhysicalUnitsData_E, PhysicalUnitsData_Ang
	Wave w_E=root:Data:tmpData:Analyser:w_E
	Wave w_Ang=root:Data:tmpData:Analyser:w_Ang
	Variable Nb_Erow=DimSize(w_E, 0)
	Variable Nb_Angrow=DimSize(w_Ang, 0)
//	PhysicalUnitsData_E[0, Nb_Erow-1] = w_E(p)
//	PhysicalUnitsData_Ang[0, Nb_Angrow-1] = w_Ang(p)
//	PhysicalUnitsData_E[Nb_Erow] = 2*PhysicalUnitsData_E(Nb_Erow-1)-PhysicalUnitsData_E(Nb_Erow-2)
//	PhysicalUnitsData_Ang[Nb_Angrow] = 2*PhysicalUnitsData_Ang(Nb_Angrow-1)-PhysicalUnitsData_Ang(Nb_Angrow-2)

	Duplicate/O PhysicalUnitsData ARPESData
	NVAR EkinLow = root:Data:tmpData:Analyser:EkinLow
	NVAR EkinHigh = root:Data:tmpData:Analyser:EkinHigh
	NVAR AzimuthLow = root:Data:tmpData:Analyser:AzimuthLow
	NVAR AzimuthHigh = root:Data:tmpData:Analyser:AzimuthHigh
//	SetScale/I x EkinLow,EkinHigh,"", ARPESData
//	SetScale/I y AzimuthLow,AzimuthHigh,"", ARPESData
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function Button_EdgeCorrection(ctrlName) : ButtonControl
	String ctrlName

	NVAR Edge_Correction=root:Data:tmpData:Analyser:Edge_Correction
	NVAR Edge_Slope=root:Data:tmpData:Analyser:Edge_Slope
	
	Variable i

	Scaling_Correction()

	SetDataFolder root:Data:tmpData:
	Wave RawData, PhysicalUnitsData
	PhysicalUnits_Data(RawData, PhysicalUnitsData)
	Duplicate/O PhysicalUnitsData ARPESData

	NVAR EkinLow = root:Data:tmpData:Analyser:EkinLow
	NVAR EkinHigh = root:Data:tmpData:Analyser:EkinHigh
	NVAR AzimuthLow = root:Data:tmpData:Analyser:AzimuthLow
	NVAR AzimuthHigh = root:Data:tmpData:Analyser:AzimuthHigh
//	SetScale/I x EkinLow,EkinHigh,"", ARPESData
//	SetScale/I y AzimuthLow,AzimuthHigh,"", ARPESData
End


Function Button_FilterImage(ctrlName) : ButtonControl
	String ctrlName
	 
	NVAR Binning=root:Data:tmpData:Analyser:Binning

	SetDataFolder root:Data:tmpData:
	Wave RawData
	
	if (strsearch(note(RawData), "FFTFiltered", 0) != -1)
		// already filtered
		return 0
	endif
	
	duplicate/o RawData, RawDataOrig

	NVAR fx=root:Data:tmpData:Analyser:filter_fx
	NVAR fy=root:Data:tmpData:Analyser:filter_fy
	NVAR wx=root:Data:tmpData:Analyser:filter_wx
	NVAR wy=root:Data:tmpData:Analyser:filter_wy
	NVAR A=root:Data:tmpData:Analyser:filter_A
	filt2dfancy(RawData, fx*(binning), fy*(binning), wx*(binning), wy*(binning), A)

End

Function Button_ResetImage(ctrlName) : ButtonControl
	String ctrlName
	 

	SetDataFolder root:Data:tmpData:
	Wave RawDataOrig

	if (waveExists(RawDataOrig))
		duplicate/o RawDataOrig, RawData
	endif

End

Function Button_PlayAround(ctrlName) : ButtonControl
	String ctrlName
	 
	
	SliderProcFourierFilter("",0,1)

	FourierFilter_PlayAround()

End


Function SetVarProcFourierFilter(ctrlName,varNum,varStr,varName) : SetVariableControl
	String ctrlName
	Variable varNum
	String varStr
	String varName

	SliderProcFourierFilter("",0,1)

End


Function SliderProcFourierFilter(ctrlName,sliderValue,event) : SliderControl
	String ctrlName
	Variable sliderValue
	Variable event	// bit field: bit 0: value set, 1: mouse down, 2: mouse up, 3: mouse moved

	if(event %& 0x1)	// bit 0, value set
	string sdf = getDataFolder(1)
	setDataFolder root:Data:tmpData

	duplicate/o RawData, temp
	NVAR fx=root:Data:tmpData:Analyser:filter_fx
	NVAR fy=root:Data:tmpData:Analyser:filter_fy
	NVAR wx=root:Data:tmpData:Analyser:filter_wx
	NVAR wy=root:Data:tmpData:Analyser:filter_wy
	NVAR A=root:Data:tmpData:Analyser:filter_A
	NVAR binning=root:Data:tmpData:Analyser:Binning
	
	filt2dfancy(temp, fx*binning, fy*binning, wx*binning, wy*binning, A, keepFFT=1)	 
	
	wave filter
	duplicate/o filter, filter_mag
	redimension/R filter_mag
	filter_mag = real(r2polar(filter))

	setDataFolder $sdf
	endif

	return 0
End

Function FourierFilter_PlayAround()
	PauseUpdate; Silent 1		// building window...
	DoWindow/K w_FourierFilter_PlayAround
	Display /W=(159,101.75,760,554) as "w_FourierFilter_PlayAround"
	DoWindow/C w_FourierFilter_PlayAround
	AppendMatrixContour/T/L=l2 root:Data:tmpData:filter_mag
	ModifyContour filter_mag labels=0
	AppendImage/T root:Data:tmpData:imageFT
	ModifyImage imageFT ctab= {0,5190920,Rainbow256,0}
	AppendImage/T/L=l2 root:Data:tmpData:imageFT_FILT
	ModifyImage imageFT_FILT ctab= {0,5190920,Rainbow256,0}
	AppendImage/B=b2/R root:Data:tmpData:temp
	ModifyImage temp ctab= {*,*,Terrain256,0}
	AppendImage/B=b2/R=r2 root:Data:tmpData:RawData
	ModifyImage RawData ctab= {*,*,Terrain256,0}
	NewFreeAxis/O/B b2_dup0
	ModifyFreeAxis/Z b2_dup0,master= b2
	ModifyGraph margin(left)=14,margin(bottom)=14,margin(top)=14,margin(right)=14
	ModifyGraph tick(left)=2,tick(top)=2,tick(l2)=2,tick(b2_dup0)=1,tick(right)=2,tick(b2)=2
	ModifyGraph tick(r2)=2
	ModifyGraph mirror(left)=0,mirror(top)=1,mirror(l2)=1,mirror(right)=0
	ModifyGraph nticks(left)=6,nticks(top)=4
	ModifyGraph minor=1
	ModifyGraph noLabel(b2_dup0)=2
	ModifyGraph fSize=8
	ModifyGraph standoff(left)=0,standoff(top)=0,standoff(l2)=0,standoff(right)=0,standoff(b2)=0
	ModifyGraph standoff(r2)=0
	ModifyGraph axThick=0.75
	ModifyGraph lblPosMode(left)=3,lblPosMode(top)=3,lblPosMode(l2)=3,lblPosMode(right)=3
	ModifyGraph lblPosMode(b2)=3,lblPosMode(r2)=3
	ModifyGraph lblPos=25
	ModifyGraph tkLblRot(left)=90
	ModifyGraph btLen=4
	ModifyGraph tlOffset(left)=-2,tlOffset(top)=-2
	ModifyGraph freePos(l2)=0
	ModifyGraph freePos(b2_dup0)={0.5,kwFraction}
	ModifyGraph freePos(b2)=0
	ModifyGraph freePos(r2)=0
	ModifyGraph axisEnab(left)={0.5,1}
	ModifyGraph axisEnab(top)={0,0.48}
	ModifyGraph axisEnab(l2)={0,0.5}
	ModifyGraph axisEnab(b2_dup0)={0.52,1}
	ModifyGraph axisEnab(right)={0,0.5}
	ModifyGraph axisEnab(b2)={0.52,1}
	ModifyGraph axisEnab(r2)={0.5,1}
	ControlBar 80
	Slider sliderA,pos={13,26},size={150,52},proc=SliderProcFourierFilter
	Slider sliderA,limits={0,1,0},variable= root:Data:tmpData:Analyser:filter_A,vert= 0
	Slider sliderfx,pos={161,26},size={150,63},proc=SliderProcFourierFilter
	Slider sliderfx,limits={0,0.1,0},variable= root:Data:tmpData:Analyser:filter_fx,vert= 0
	Slider sliderfy,pos={311,24},size={150,63},proc=SliderProcFourierFilter
	Slider sliderfy,limits={0,0.1,0},variable= root:Data:tmpData:Analyser:filter_fy,vert= 0
	Slider sliderfx3,pos={454,24},size={150,63},proc=SliderProcFourierFilter
	Slider sliderfx3,limits={0,0.02,0},variable= root:Data:tmpData:Analyser:filter_wx,vert= 0
	Slider sliderwy,pos={603,25},size={150,63},proc=SliderProcFourierFilter
	Slider sliderwy,limits={0,0.02,0},variable= root:Data:tmpData:Analyser:filter_wy,vert= 0
	SetVariable FilterwA,pos={23,8},size={78,16},bodyWidth=62,proc=SetVarProcFourierFilter,title="A:"
	SetVariable FilterwA,fStyle=1
	SetVariable FilterwA,limits={0,1,0.01},value= root:Data:tmpData:Analyser:filter_A
	SetVariable FilterfX,pos={187,8},size={82,16},bodyWidth=62,proc=SetVarProcFourierFilter,title="fX:"
	SetVariable FilterfX,fStyle=1
	SetVariable FilterfX,limits={0,0.1,0.001},value= root:Data:tmpData:Analyser:filter_fx
	SetVariable FilterwX,pos={479,9},size={87,16},bodyWidth=62,proc=SetVarProcFourierFilter,title="wX:"
	SetVariable FilterwX,fStyle=1
	SetVariable FilterwX,limits={0,0.02,0.0002},value= root:Data:tmpData:Analyser:filter_wX
	SetVariable FilterwY,pos={636,8},size={87,16},bodyWidth=62,proc=SetVarProcFourierFilter,title="wY:"
	SetVariable FilterwY,fStyle=1
	SetVariable FilterwY,limits={0,0.02,0.0002},value= root:Data:tmpData:Analyser:filter_wY
	SetVariable FilterfY,pos={347,6},size={82,16},bodyWidth=62,proc=SetVarProcFourierFilter,title="fY:"
	SetVariable FilterfY,fStyle=1
	SetVariable FilterfY,limits={0,0.1,0.001},value= root:Data:tmpData:Analyser:filter_fY
	Button buttonExit,pos={743,4},size={50,20},proc=FourierFilter_PA_Exit,title="Exit"
	Button buttonExit,fSize=14,fStyle=1,fColor=(0,65280,0)
EndMacro


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function FourierFilter_PA_Exit(ctrlName) : ButtonControl
	String ctrlName
	
	SetDataFolder root:Data:tmpData:
	DoWindow/K w_FourierFilter_PlayAround
	killwaves/Z tmp_image, LowPass, filter, imageFT, imageFT_FILT
End

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function ImportData_Setting_Exit(ctrlName) : ButtonControl
	String ctrlName
	
	SetDataFolder root:Data:tmpData:
	DoWindow/K Gr_ImportData_Setting
	KillWaves/Z RawData, PhysicalUnitsData, ARPESData, CropLine_x, CropLine
	KillWaves/Z PhysicalUnitsData_E, PhysicalUnitsData_Ang
End

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Window Gr_ImportData_Setting() : Graph
	PauseUpdate; Silent 1		// building window...
	String fldrSav0= GetDataFolder(1)
	SetDataFolder root:Data:tmpData:
	Display /W=(47.25,242.75,814.5,704.75)/L=Left_PhysicalUnit/T CropLine_x vs CropLine as "Settings to import Tr-ARPES Data"
	AppendImage RawData
	ModifyImage RawData ctab= {*,*,Terrain256,0}
	AppendImage/T/L=Left_PhysicalUnit PhysicalUnitsData// vs {PhysicalUnitsData_E,PhysicalUnitsData_Ang}
	ModifyImage PhysicalUnitsData ctab= {*,*,Terrain256,0}
	AppendImage/T/L=Left_ARPES ARPESData
	ModifyImage ARPESData ctab= {*,*,Terrain256,0}
	SetDataFolder fldrSav0
	ModifyGraph rgb=(65535,65535,65535)
	ModifyGraph mirror(Left_PhysicalUnit)=1,mirror(top)=0,mirror(left)=1,mirror(bottom)=0
	ModifyGraph mirror(Left_ARPES)=1
	ModifyGraph minor(top)=1
	ModifyGraph fSize=12
	ModifyGraph fStyle=1
	ModifyGraph lblMargin(top)=1,lblMargin(bottom)=3
	ModifyGraph standoff=0
	ModifyGraph axOffset(top)=-2.42857,axOffset(bottom)=-2.22222
	ModifyGraph axThick=2
	ModifyGraph lblPos(Left_PhysicalUnit)=51,lblPos(left)=51,lblPos(Left_ARPES)=51
	ModifyGraph lblLatPos(Left_ARPES)=2
	ModifyGraph freePos(Left_PhysicalUnit)=0
	ModifyGraph freePos(Left_ARPES)=0
	ModifyGraph axisEnab(Left_PhysicalUnit)={0.34,0.65}
	ModifyGraph axisEnab(left)={0,0.32}
	ModifyGraph axisEnab(Left_ARPES)={0.67,1}
	Label Left_PhysicalUnit "Angle (deg)"
	Label top "Kinetic Energy (eV)"
	Label left "Angle (pixel)"
	Label bottom "Kinetic Energy (pixel)"
	Label Left_ARPES "Angle (deg)"
	//SetAxis Left_ARPES -15.0579150579151,15.0579150579151
	Cursor/P/I/H=2 A RawData 23,126;Cursor/P/I/H=2 B PhysicalUnitsData 173,226;Cursor/P/I/S=2/H=1 C PhysicalUnitsData 22,40;Cursor/P/I/S=2/H=1 D PhysicalUnitsData 319,225
	ShowInfo
	ControlBar 88
	GroupBox group3,pos={688,2},size={203,83}
	GroupBox group2,pos={574,2},size={107,83}
	GroupBox group1,pos={419,2},size={148,83}
	GroupBox group0,pos={159,2},size={253,83}
	SetVariable D_E1,pos={799,40},size={87,16},bodyWidth=60,title="E1 :",fStyle=1
	SetVariable D_E1,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:E1
	SetVariable D_E2,pos={799,23},size={87,16},bodyWidth=60,title="E2 :",fStyle=1
	SetVariable D_E2,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:E2
	SetVariable D_Theta1,pos={694,39},size={101,16},bodyWidth=60,title="Ang1 :"
	SetVariable D_Theta1,fStyle=1
	SetVariable D_Theta1,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:Theta1
	SetVariable D_Theta2,pos={694,22},size={101,16},bodyWidth=60,title="Ang2 :"
	SetVariable D_Theta2,fStyle=1
	SetVariable D_Theta2,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:Theta2
	SetVariable D_nSize,pos={581,24},size={91,16},bodyWidth=32,title="n*n Size :"
	SetVariable D_nSize,fStyle=1
	SetVariable D_nSize,limits={1,101,1},value= root:Data:tmpData:Analyser:n_size
	SetVariable D_passes,pos={589,41},size={83,16},bodyWidth=32,title="passes :"
	SetVariable D_passes,fStyle=1
	SetVariable D_passes,limits={1,101,1},value= root:Data:tmpData:Analyser:passes
	Button D_TestSmoothing,pos={585,60},size={85,22},proc=Button_SmoothingImage,title="Smooth"
	Button D_TestSmoothing,fStyle=1
	Button button0,pos={971,4},size={50,20},proc=ImportData_Setting_Exit,title="Exit"
	Button button0,fSize=14,fStyle=1,fColor=(0,65280,0)
	Button D_TestCropImage1,pos={708,57},size={69,22},proc=Button_CropImage,title="Crop"
	Button D_TestCropImage1,fStyle=1
	Button LoadExternalData,pos={893,26},size={120,20},proc=LoadExternalImage,title="Load Image ..."
	Button LoadExternalData,fStyle=1
	Button LoadTestData,pos={893,45},size={120,20},proc=LoadFirstImage,title="Auto Data ..."
	Button LoadTestData,fStyle=1
	Button LoadDefaultParameters,pos={893,64},size={120,20},proc=DefaultParametersFrom_Calib2D,title="Default Parameters"
	Button LoadDefaultParameters,fStyle=1
	PopupMenu Popup_LensMode,pos={168,5},size={188,21},title="Lens Mode",fStyle=1
	PopupMenu Popup_LensMode,mode=4,popvalue="Wide Angular Mode",value= #"\"Low Angular Dispersion;Medium Angular Dispersion;High Angular Dispersion;Wide Angular Mode\""
	SetVariable KineticEnergy,pos={164,28},size={104,16},bodyWidth=50,title="Ek (eV) :"
	SetVariable KineticEnergy,fStyle=1
	SetVariable KineticEnergy,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:Ek
	SetVariable PassEnergy,pos={164,46},size={104,16},bodyWidth=50,title="Ep (eV) :"
	SetVariable PassEnergy,fStyle=1
	SetVariable PassEnergy,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:Ep
	//Claude:22.11.13
	SetVariable RotationAngle,pos={268,46},size={127,16},bodyWidth=35,title="Rotation (°) :"
	SetVariable RotationAngle,fStyle=1
	SetVariable RotationAngle,limits={-inf,inf,1},value= root:Data:tmpData:Analyser:Rotation_Angle
	PopupMenu popup_Binning,pos={304,24},size={98,16},bodyWidth=45,title="Binning :"
	PopupMenu popup_Binning,fStyle=1, proc=popupMenu_Binning
	PopupMenu popup_Binning,mode=1,popvalue="4x",value= #"\"1x;2x;4x;8x;16x\""
	Button PhysicalUnits,pos={325,62},size={80,20},proc=Button_PhysicalUnits,title="Ang. Correc."
	Button PhysicalUnits,fStyle=1
	SetVariable Ang_Offset_px,pos={184,64},size={136,16},bodyWidth=36,title="Ang Centre (px) :"
	SetVariable Ang_Offset_px,fStyle=1
	SetVariable Ang_Offset_px,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:Ang_Centre_px
	Button EdgeCorrection,pos={458,62},size={80,20},proc=Button_EdgeCorrection,title="Edge Correc."
	Button EdgeCorrection,fStyle=1
	SetVariable Edge_pos,pos={426,27},size={136,16},bodyWidth=42,title="Edge pos(deg) :"
	SetVariable Edge_pos,fStyle=1
	SetVariable Edge_pos,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:Edge_pos
	SetVariable Edge_Slope,pos={443,44},size={119,16},bodyWidth=42,title="Edge Slope :"
	SetVariable Edge_Slope,fStyle=1
	SetVariable Edge_Slope,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:Edge_Slope
	CheckBox CheckEdgeCorrection,pos={423,7},size={109,14},title="Edge Correction"
	CheckBox CheckEdgeCorrection,fStyle=1
	CheckBox CheckEdgeCorrection,variable= root:Data:tmpData:Analyser:Edge_Correction
	CheckBox CropWithCursor,pos={798,61},size={80,14},title="Use Cursor",fStyle=1
	CheckBox CropWithCursor,value= 1
	CheckBox CheckSmooth,pos={578,7},size={80,14},title="2D Smooth",fStyle=1
	CheckBox CheckSmooth,variable= root:Data:tmpData:Analyser:Check_2D_Smooth
	CheckBox CheckCrop,pos={691,7},size={44,14},title="Crop",fStyle=1
	CheckBox CheckCrop,variable= root:Data:tmpData:Analyser:Crop
	// Laurenz 12.7.16
	GroupBox group4,pos={5,2},size={148,83}
	CheckBox CheckFilter,variable= root:Data:tmpData:Analyser:Check_2D_Smooth
	CheckBox CheckFilter,pos={9,7},size={44,14},title="Fourier Filter",fStyle=1
	CheckBox CheckFilter,variable= root:Data:tmpData:Analyser:Filter_Image
	SetVariable FilterfX,pos={11, 25},size={62,16},bodyWidth=42,title="fX:"
	SetVariable FilterfX,fStyle=1
	SetVariable FilterfX,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:filter_fx
	SetVariable FilterfY,pos={11, 43},size={62,16},bodyWidth=42,title="fY:"
	SetVariable FilterfY,fStyle=1
	SetVariable FilterfY,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:filter_fY
	SetVariable FilterwX,pos={80, 25},size={62,16},bodyWidth=42,title="wX:"
	SetVariable FilterwX,fStyle=1
	SetVariable FilterwX,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:filter_wX
	SetVariable FilterwY,pos={80, 43},size={62,16},bodyWidth=42,title="wY:"
	SetVariable FilterwY,fStyle=1
	SetVariable FilterwY,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:filter_wY
	SetVariable FilterwA,pos={105, 6},size={42,16},bodyWidth=25,title="A:"
	SetVariable FilterwA,fStyle=1
	SetVariable FilterwA,limits={-inf,inf,0},value= root:Data:tmpData:Analyser:filter_A
	Button FilterImage,pos={10,62},size={40,20},proc=Button_FilterImage,title="Filter"
	Button FilterImage,fStyle=1
	Button ResetImage,pos={60,62},size={40,20},proc=Button_ResetImage,title="Reset"
	Button ResetImage,fStyle=1	
	Button PlayAround,pos={110,62},size={40,20},proc=Button_PlayAround,title="Play"
	Button PlayAround,fStyle=1	
	
	ModifyGraph swapXY=1
EndMacro


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function popupMenu_Binning(ctrlName,popNum,popStr, [set_binning]) : PopupMenuControl
	String ctrlName
	Variable popNum
	String popStr	
	variable set_binning
	
	NVAR binning = root:Data:tmpData:Analyser:binning
	if (ParamIsDefault(set_binning))
		// get binning from popup menu
		binning = 2^(popNum-1)
	else
		binning = set_binning
		// set control
		PopupMenu Popup_Binning win=Gr_ImportData_Setting, mode=log(binning)/log(2)+1
	endif		
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Smooth
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function Button_SmoothingImage(ctrlName) : ButtonControl
	String ctrlName

	SetDataFolder root:Data:tmpData:

	Wave PhysicalUnitsData
	Duplicate/O PhysicalUnitsData ARPESData
	NVAR EkinLow = root:Data:tmpData:Analyser:EkinLow
	NVAR EkinHigh = root:Data:tmpData:Analyser:EkinHigh
	NVAR AzimuthLow = root:Data:tmpData:Analyser:AzimuthLow
	NVAR AzimuthHigh = root:Data:tmpData:Analyser:AzimuthHigh
//	SetScale/I x EkinLow,EkinHigh,"", ARPESData
//	SetScale/I y AzimuthLow,AzimuthHigh,"", ARPESData

	Make/O/N=5 CropLine_x=NaN, CropLine=NaN

	SmoothingImage(ARPESData)

end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function SmoothingImage_old(Image)
Wave Image


	NVAR n_size=root:Data:tmpData:Analyser:n_size
	NVAR passes=root:Data:tmpData:Analyser:passes


	MatrixFilter/N=(n_size)/P=(passes) Gauss, Image

end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//Boxcar Smoothing
function SmoothingImage(Image)
Wave Image


	NVAR n_size=root:Data:tmpData:Analyser:n_size
	NVAR passes=root:Data:tmpData:Analyser:passes


	Smooth/B=(passes)/Dim=0 n_size, Image
	Smooth/B=(passes)/Dim=1 n_size, Image

end


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Crop
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function Button_CropImage(ctrlName) : ButtonControl
	String ctrlName

	SetDataFolder root:Data:tmpData:Analyser

	NVAR E1=root:Data:tmpData:Analyser:E1
	NVAR E2=root:Data:tmpData:Analyser:E2
	NVAR Theta1=root:Data:tmpData:Analyser:Theta1
	NVAR Theta2=root:Data:tmpData:Analyser:Theta2

	Button_SmoothingImage("")

	Wave ARPESData=root:Data:tmpData:ARPESData
	
	ControlInfo CropWithCursor	// Use Cursor?
	if ( V_value==1 )
		E1=min(hcsr(C), hcsr(D))
		E2=max(hcsr(C), hcsr(D))
		
		Theta1=min(vcsr(C), vcsr(D))
		Theta2=max(vcsr(C), vcsr(D))		
	endif

	Make/O/N=5 CropLine_x={Theta1, Theta2, Theta2, Theta1, Theta1}
	Make/O/N=5 CropLine={E1, E1, E2, E2, E1}

//	GetAxis/Q Left_ARPES
//	SetAxis/Z Left_ARPES, V_Min, V_Max

	CropParameters(ARPESData)
	Image_Crop(ARPESData)
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function CropParameters(Image)
Wave Image

	String fldrSav0= GetDataFolder(1)
	SetDataFolder root:Data:tmpData:Analyser:

	NVAR E1, E2, Theta1, Theta2

	Variable Max_nE, Max_nTheta
	Variable/G Delta_E, Delta_Theta, Delete_nE, Delete_nTheta, NewOffset_E, NewOffset_Theta, NewDim_E, NewDim_Theta
	
	Max_nE=DimSize(Image, 0)
	Max_nTheta=DimSize(Image, 1)

	Delta_E=DimDelta(Image,0)
	Delta_Theta=DimDelta(Image,1)

	Delete_nE= limit( round((E1 - DimOffset(Image, 0))/Delta_E) , 0, Max_nE)
	Delete_nTheta= limit( round((Theta1 - DimOffset(Image, 1))/Delta_Theta) , 0, Max_nTheta)

	NewOffset_E=DimOffset(Image, 0)+Delta_E*Delete_nE
	NewOffset_Theta=DimOffset(Image, 1)+Delta_Theta*Delete_nTheta

	NewDim_E=limit( round((E2-NewOffset_E)/Delta_E+1), 1, Max_nE)
	NewDim_Theta=limit( round((Theta2-NewOffset_Theta)/Delta_Theta+1), 1, Max_nTheta)

	SetDataFolder fldrSav0
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function Image_Crop(Image)
Wave Image

	NVAR Delete_nE=root:Data:tmpData:Analyser:delete_nE
	NVAR Delete_nTheta=root:Data:tmpData:Analyser:Delete_nTheta
	NVAR NewDim_E=root:Data:tmpData:Analyser:NewDim_E
	NVAR NewDim_Theta=root:Data:tmpData:Analyser:NewDim_Theta
	NVAR NewOffset_E=root:Data:tmpData:Analyser:NewOffset_E
	NVAR NewOffset_Theta=root:Data:tmpData:Analyser:NewOffset_Theta
	NVAR Delta_E=root:Data:tmpData:Analyser:Delta_E
	NVAR Delta_Theta=root:Data:tmpData:Analyser:Delta_Theta

	
	DeletePoints/M=0 0,Delete_nE, Image
	DeletePoints/M=1 0,Delete_nTheta, Image
	
	Redimension/N=(NewDim_E,NewDim_Theta) Image
	
	SetScale/P x NewOffset_E,Delta_E,"", Image
	SetScale/P y NewOffset_Theta,Delta_Theta,"", Image
	
end


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////// PHYSICAL UNIT CONVERSION /////////////////////////////////////////////////////////////////////

//~~~~~~~~~~~~~~~~~All the parameters needed to do the angular correction
//	LensMode			from user
//	Ek					from user
//	Ep					from user
//	E_Offset_px			from user
//	Ang_Offset_px		from user
//	Binning				from user
//	Edge_pos			from user
//	Edge_Slope			from user
//	eShift				from phoibos100.Calib2D
//	eRange				from phoibos100.Calib2D
//	aRange				from phoibos100.Calib2D
//	De1					from phoibos100.Calib2D
//	aInner				from phoibos100.Calib2D
//	Da3 - Da7			from phoibos100.Calib2D
//	WF					4.2eV or from CCDAcquire
//	PixelSize			0.00645 or from CCDAcquire
//	magnification			4.54 or from CCDAcquire
//~~~~~~~~~~~~~~~~~All the parameters needed to do the angular correction



// External function for doing angular and energy conversion
// Creates output wave Data in current data folder
function/S ConvertImage(w, Epass, Ekin, LensMode_, binning_, [Udet])
	wave w
	variable Epass, Ekin, LensMode_, binning_, Udet
	NVAR LensMode = root:Data:tmpData:Analyser:LensMode
	NVAR binning = root:Data:tmpData:Analyser:binning
	NVAR Ek = root:Data:tmpData:Analyser:Ek
	NVAR Ep = root:Data:tmpData:Analyser:Ep
	NVAR CCDcounts2ecounts = root:Data:tmpData:Analyser:CCDcounts2ecounts
	Ek=Ekin
	Ep=Epass
	LensMode=LensMode_
	binning = binning_
	
	if (ParamIsDefault(Udet))
		CCDcounts2ecounts = 1
	else
		// take 3rd order polynomial fit to counts/hit calibration. See Matlab analysis
		CCDcounts2ecounts = 1/(1.4*(8.4077E-5*(Udet+1300)^3 + 0.00094145*(Udet+1300)^2 + 7.9855*(Udet+1300) + 255.4808))
	endif
	
	Wave Angular_Correction=root:Data:tmpData:Analyser:Angular_Correction
	NVAR Edge_Correction=root:Data:tmpData:Analyser:Edge_Correction
	if ( (NumberByKey("LensMode",Note(Angular_Correction))!=LensMode) || (NumberByKey("Ek",Note(Angular_Correction))!=Ek) || (NumberByKey("PE",Note(Angular_Correction))!=Ep) || (NumberByKey("binning",Note(Angular_Correction))!=binning))
		string DF = getDataFolder(1)
		Calculate_Da_values()
		Calculate_Polynomial_Coef_Da()
		Calculate_MatrixCorrection()
		print "New Angular Correction Matrix"	
	
		if (Edge_Correction)
			String fldrSav0= GetDataFolder(1)
			Scaling_Correction()
			SetDataFolder fldrSav0
		endif
		
		NVAR Crop = root:Data:tmpData:Analyser:Crop
		if (Crop==1)
			Crop = 0
			print "Crop has been disabled, revisit Settings window to enable!"
		endif
		setDataFolder $DF
	endif
	NVAR EkinLow = root:Data:tmpData:Analyser:EkinLow
	NVAR EkinHigh = root:Data:tmpData:Analyser:EkinHigh
	NVAR AzimuthLow = root:Data:tmpData:Analyser:AzimuthLow
	NVAR AzimuthHigh = root:Data:tmpData:Analyser:AzimuthHigh
	NVAR Filter_Image=root:Data:tmpData:Analyser:Filter_Image
	NVAR Crop=root:Data:tmpData:Analyser:Crop
	NVAR Check_2D_Smooth=root:Data:tmpData:Analyser:Check_2D_Smooth
	
	if ( DimSize(w,0) < DimSize(w, 1) )		// TransposeMatrix in (E, Theta) the orientation of the Data matrix
		duplicate/o w, w_temp
		MatrixTranspose w_temp
		wave w = w_temp
	endif
	string wname_out = getWavesDataFolder(w, 1) + nameofwave(w) + "_conv"
	if ( Crop==1 )
		NVAR Delete_nE=root:Data:tmpData:Analyser:delete_nE
		NVAR Delete_nTheta=root:Data:tmpData:Analyser:Delete_nTheta
		NVAR NewDim_E=root:Data:tmpData:Analyser:NewDim_E
		NVAR NewDim_Theta=root:Data:tmpData:Analyser:NewDim_Theta
		NVAR NewOffset_E=root:Data:tmpData:Analyser:NewOffset_E
		NVAR NewOffset_Theta=root:Data:tmpData:Analyser:NewOffset_Theta
		NVAR Delta_E=root:Data:tmpData:Analyser:Delta_E
		NVAR Delta_Theta=root:Data:tmpData:Analyser:Delta_Theta

		Make/O/N=(NewDim_E, NewDim_Theta) $wname_out	
		wave Data = $wname_out
		//Make/O/N=(NewDim_E+1) Data_E
		//Make/O/N=(NewDim_Theta+1) Data_k
		// Internal wave scaling
		Setscale/P x, dimOffset(Angular_Correction,0)+dimDelta(Angular_Correction,0)*Delete_nE, dimDelta(Angular_Correction,0), waveUnits(Angular_Correction,0), Data
		Setscale/P y, dimOffset(Angular_Correction,1)+dimDelta(Angular_Correction,1)*Delete_nTheta, dimDelta(Angular_Correction, 1), waveUnits(Angular_Correction,1), Data

		//Data_E[]=NewOffset_E+p*Delta_E
		//Data_k[]=NewOffset_Theta+p*Delta_Theta
		
	else
		Make/O/N=(DimSize(w, 0), DimSize(w, 1)) $wname_out
		wave Data = $wname_out
		//Make/O/N=(DimSize(LoadW0, 0)+1) Data_E
		//Make/O/N=(DimSize(LoadW0, 1)+1) Data_k
		// Internal wave scaling
		Setscale/P x, dimOffset(Angular_Correction,0), dimDelta(Angular_Correction,0), waveUnits(Angular_Correction,0), Data
		Setscale/P y, dimOffset(Angular_Correction,1), dimDelta(Angular_Correction,1), waveUnits(Angular_Correction,1), Data
	endif
	
	if (filter_Image)
		NVAR fx=root:Data:tmpData:Analyser:filter_fx
		NVAR fy=root:Data:tmpData:Analyser:filter_fy
		NVAR wx=root:Data:tmpData:Analyser:filter_wx
		NVAR wy=root:Data:tmpData:Analyser:filter_wy
		NVAR A=root:Data:tmpData:Analyser:filter_A
		filt2dfancy(w, fx*binning, fy*binning, wx*binning, wy*binning, A)	 
	endif	
	
	duplicate/o w, w_corrected
	PhysicalUnits_Data(w, w_corrected)
				
	if ( Check_2D_Smooth )
		SmoothingImage(w_corrected)
	endif
	
	if ( Crop )
		DeletePoints/M=0 0,Delete_nE, w_corrected
		DeletePoints/M=1 0,Delete_nTheta, w_corrected
	endif
	// put it to normal scaling
	Data[][] = w_corrected[p][q]
	MatrixTranspose Data
	KillWaves/Z w_corrected, w_temp
	return getwavesDataFolder(Data, 2)
end


// Fourier Filtering of the Grid
//FILTER THE IMAGE TO REMOVE THE GRID
Function filt2dfancy(image,freqx, freqy,sigmax, sigmay,ampl, [keepFFT])
	wave image
	variable freqx, freqy,sigmax, sigmay,ampl, keepFFT
	
	if (paramIsDefault(keepFFT))
		keepFFT=0
	endif

	Duplicate/O image tmp_image
	//Redimension/s tmpimage

	//duplicate/O/I tmp_image window_image
	//window_image=1
	
	// Apply Hamming window to avoid FFT artifacts
	//ImageWindow/O hamming window_image
	//ImageWindow/O hamming tmp_image

	// Do complex Fourier Transform of the image
	FFT tmp_image
	Duplicate/O/C tmp_image lowPass // new complex wave in freq. domain
	Duplicate/O/C tmp_image filter //complex filter matrix
	Duplicate/O/C tmp_image imageFT

	variable GaussNorm = gauss(0,0,sigmax,0,0,sigmay)
	filter=cmplx(1,0)

	variable indexx, indexy
	variable freqxtmp, freqytmp

	// Put gaussian peaks at +-1 in x and y dimension in filter wave
	for(indexx=0;indexx<2;indexx=indexx+1)	
		for(indexy=-1;indexy<2;indexy=indexy+1)	
			freqxtmp=indexx*freqx
			freqytmp=indexy*freqy
			if (indexx!=0 || indexy !=0)
				filter=filter-ampl/GaussNorm*cmplx((gauss(x,freqxtmp,sigmax,y,freqytmp,sigmay)),0)
			endif
		endfor
	endfor     
	
	// filter amplitude by complex division	
	lowPass=imageFT*filter
	Duplicate/O/C lowPass imageFT_FILT

	IFFT lowPass
	// cut off negative values
	Image[][] = (lowPass[p][q] < 0) ? 0 : LowPass[p][q]
	utils_addWaveNoteEntry(image, "FFTFiltered", "1")
	//duplicate/O lowpass lowpass2
	//lowpass=lowpass/window_image
	if (!keepFFT)
		killwaves/Z tmp_image, LowPass, filter, imageFT, imageFT_FILT
	endif
	
end












Function Initialise_AngCorrection()

	DefaultParametersFrom_Calib2D("")
	Calculate_Da_values()
	Calculate_Polynomial_Coef_Da()
	
	NVAR Binning=root:Data:tmpData:Analyser:Binning
	NVAR E_Offset_px=root:Data:tmpData:Analyser:E_Offset_px
	NVAR Ang_Offset_px=root:Data:tmpData:Analyser:Ang_Offset_px
	NVAR Edge_pos=root:Data:tmpData:Analyser:Edge_pos
	Binning=4
	E_Offset_px= 0
	Ang_Offset_px=0

	Calculate_MatrixCorrection()	
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function PhysicalUnits_Data(rawData, CorrectedData)
Wave rawData, CorrectedData

	Wave Angular_Correction=root:Data:tmpData:Analyser:Angular_Correction
	Wave E_Correction=root:Data:tmpData:Analyser:E_Correction
	Wave Jacobian_Determinant=root:Data:tmpData:Analyser:Jacobian_Determinant
	
	NVAR CCDcounts2ecounts = root:Data:tmpData:Analyser:CCDcounts2ecounts
	
	//Claude: 22.11.13
	NVAR Rotation_Angle=root:Data:tmpData:Analyser:Rotation_Angle
	if(Rotation_Angle!=0)
		duplicate/o rawData, rawData_rot
		wave rawData = rawData_rot
		variable xsize=dimsize(RawData,0)
		variable ysize=dimsize(RawData,1)
		ImageRotate/A=(Rotation_angle)/E=0/O RawData
		variable newxsize=dimsize(RawData,0)
		variable newysize=dimsize(RawData,1)
		DeletePoints 0,ceil((newxsize-xsize)/2),RawData
		DeletePoints dimsize(RawData,0)-floor((newxsize-xsize)/2),floor((newxsize-xsize)/2),RawData
		DeletePoints/M=1 0,ceil((newysize-ysize)/2),RawData
		DeletePoints/M=1 dimsize(RawData,1)-floor((newysize-ysize)/2),floor((newysize-ysize)/2),RawData
	endif
	
	CorrectedData = NaN
//	CorrectedData[][]=rawData( E_Correction(p) )( Angular_Correction(p)(q) )
	// wave scaling from Angular_Correction:
	setScale/P x, dimOffset(Angular_Correction,0), dimDelta(Angular_Correction,0), waveUnits(Angular_Correction,0), CorrectedData
	setScale/P y, dimOffset(Angular_Correction,1), dimDelta(Angular_Correction,1), waveUnits(Angular_Correction,1), CorrectedData
	// proper interpolation
	// take Jacobian Determinant of transformation matrix to correct the intensity.
	//CorrectedData[][] = interp2D(rawData, E_Correction[p], Angular_Correction[p][q])
	CorrectedData[][] = CCDcounts2ecounts * Jacobian_Determinant[p][q] * interp2D(rawData, E_Correction[p], Angular_Correction[p][q])
	//MultiThread  DataCorrected[][]=Data( E_Correction(p) )( Angular_Correction(p)(q) )
	
	// fix NaN's to Zero
	CorrectedData[][] = (NumType(CorrectedData[p][q])==2) ? 0 : CorrectedData[p][q]
	
	Killwaves/Z rowData_rot
end




//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~Read Parameters from Calibration File~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function DefaultParametersFrom_Calib2D(ctrlName) : ButtonControl
	String ctrlName

	SetDataFolder root:Data:tmpData:'Analyser':

	Variable/G LensMode, Ek, Ep, WF
	Variable/G PixelSize, magnification, E_Offset_px, Ang_Offset_px, Binning=1, De1, aInner
	Variable/G EkinLow, EkinHigh, AzimuthLow, AzimuthHigh
	Variable/G Edge_pos, Edge_Slope
	Variable FileRef, numItems, line, n_rr
	String contents="", str, key="", value=""
	
	//open/R/P=artemis_procedures FileRef as "phoibos150.calib2d"  // 03.07.14: Claude: modified to 150
	//open/R/P=artemis_procedures FileRef as "phoibosEPFL.calib2d"  // 05.12.17: MICHELE_EPFL
	open/R/P=artemis_procedures FileRef as "phoibosEPFL_052018.calib2d"  // 24.05.18: MICHELE_EPFL
//	contents = PadString(contents, 80000, 0)
	contents = PadString(contents, 52474, 0)  //change here with real length of file to avoid error
	FBinRead FileRef, contents
	close FileRef
	
	numItems = ItemsInList(contents,"\n")
	Make/O/T/N=(numItems) phoibos100_Calib

	phoibos100_Calib[]=StringFromList(p,contents,"\n")
  	phoibos100_Calib[] = removeCommentChar(phoibos100_Calib(p))
	phoibos100_Calib[] = stripwhitespace(phoibos100_Calib(p))
	
//	Build matrix with all Calib parameters
//	Calib_Matrix[LenseMode][aRange, rr_index][rr_value, aInner, Da1, Da3, Da5, Da7]
//	LenseMode order = LAD MAD HAD WAM

	Make/T/O/N=(4,25,6) Calib_Matrix=""
	for( line=0 ; line < numItems ; line+=1 )

		if( isKeyValuePair( phoibos100_Calib(line) ,key,value) )
			if( cmpstr(key,"eShift") == 0 )
				Make/O/N=3 eShift
				eShift[0]=str2num(StringFromList(0,value," "))
				eShift[1]=str2num(StringFromList(1,value," "))
				eShift[2]=str2num(StringFromList(2,value," "))						
			elseif( cmpstr(key,"eRange") == 0 )
				Make/O/N=2 eRange
				eRange[0]=str2num(StringFromList(0,value," "))
				eRange[1]=str2num(StringFromList(1,value," "))
			elseif( cmpstr(key,"de1") == 0 )
				de1 = Str2Num(value)
			endif
		elseif( cmpstr(phoibos100_Calib(line), "[LowAngularDispersion defaults]") == 0 )
			n_rr=0
			if( isKeyValuePair( phoibos100_Calib(line+2) ,key,value) )
					if( cmpstr(key,"aRange") == 0 )
						Calib_Matrix[0][0][0] = value
						line+=2
					endif
			endif
		
		elseif( cmpstr(phoibos100_Calib(line), "[WideAngleMode defaults]") == 0 )
			n_rr=0
			if( isKeyValuePair( phoibos100_Calib(line+2) ,key,value) )
					if( cmpstr(key,"aRange") == 0 )
						Calib_Matrix[3][0][0] = value
						line+=2
					endif
			endif
		
		elseif( cmpstr((phoibos100_Calib(line))[0,21], "[LowAngularDispersion@") == 0 )
			n_rr += 1
			Calib_Matrix[0][n_rr][0] = (phoibos100_Calib(line))[22,100]
			if( isKeyValuePair( phoibos100_Calib(line+1) ,key,value) )
				if( cmpstr(key,"aInner") == 0 )
					Calib_Matrix[0][n_rr][1] = value
					isKeyValuePair( phoibos100_Calib(line+2) ,key,value)
					Calib_Matrix[0][n_rr][2] = value
					isKeyValuePair( phoibos100_Calib(line+3) ,key,value)
					Calib_Matrix[0][n_rr][3] = value
					isKeyValuePair( phoibos100_Calib(line+4) ,key,value)
					Calib_Matrix[0][n_rr][4] = value
					isKeyValuePair( phoibos100_Calib(line+5) ,key,value)
					Calib_Matrix[0][n_rr][5] = value
					line+=5
				endif
			endif
		
		elseif( cmpstr((phoibos100_Calib(line))[0,14], "[WideAngleMode@") == 0 )
			n_rr += 1
			Calib_Matrix[3][n_rr][0] = (phoibos100_Calib(line))[15,100]
			if( isKeyValuePair( phoibos100_Calib(line+1) ,key,value) )
				if( cmpstr(key,"aInner") == 0 )
					Calib_Matrix[3][n_rr][1] = value
					isKeyValuePair( phoibos100_Calib(line+2) ,key,value)
					Calib_Matrix[3][n_rr][2] = value
					isKeyValuePair( phoibos100_Calib(line+3) ,key,value)
					Calib_Matrix[3][n_rr][3] = value
					isKeyValuePair( phoibos100_Calib(line+4) ,key,value)
					Calib_Matrix[3][n_rr][4] = value
					isKeyValuePair( phoibos100_Calib(line+5) ,key,value)
					Calib_Matrix[3][n_rr][5] = value
					line+=5
				endif
			endif
		endif

	endfor

	KillWaves/Z phoibos100_Calib
end
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
static Function isCommentLine(str)
  string str
  
  variable ret = cmpstr(str[0],"#")
  
  if( ret == 0)
    return 1 // is comment line
  else
    return 0 // is no comment line
  endif
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
static Function isGroup(str,group)
  string str, &group
  
  variable posStart, posEnd
  
  posStart = strsearch(str,"[",0)
  posEnd = strsearch(str,"]",Inf,1)
  
  if( posStart == 0 && posEnd != -1 )
    group = str[posStart+1,posEnd-1]
    return 1
  else
    group = ""
    return 0
  endif
 end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function/S removeCommentChar(str)
  string str
  
  if(isCommentLine(str))
    return str[1,strlen(str)]
  else
    return str
  endif
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function isKeyValuePair(str,key,value)
  string str, &key, &value

  variable posStart
  
  posStart = strsearch(str,"=",0)
  
  if( posStart != -1 )
    
    key = str[0,posStart-1]
    value = str[posStart+1,strlen(str)]
    
    key = stripwhitespace(key)
    value = stripwhitespace(value)
    value = stripStringLimiters(value)
      
    return 1
  else
    return 0
  endif
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
static Function/S stripStringLimiters(str)
  string str
  
  variable posStart, posEnd
  posStart = strSearch(str,"\"",0)
  posEnd  = strSearch(str,"\"",Inf,1)
  
  if( posStart != -1 && posEnd != 1 & posEnd != posStart )
    return str[posStart+1,posEnd-1]
  else
    return str
  endif
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function/S stripwhitespace(str)
    string str
    
    variable i=0,j=0, length=strlen(str)
    for( i=0 ; i< length; i+=1)
      if( cmpstr(str[i]," ") != 0 && cmpstr(str[i],"\t") != 0 && cmpstr(str[i],"\r") != 0 && cmpstr(str[i],"\n") != 0 )
        break
      endif
    endfor
    
    for( j=length-1 ; j != 0; j-=1)
      if( cmpstr(str[j]," ") != 0 && cmpstr(str[j],"\t") != 0 && cmpstr(str[j],"\r") != 0 && cmpstr(str[j],"\n") != 0 )
        break
      endif
    endfor  
    
    str=str[i,j]
    return str
end
//~~~~~~~~~~~~~End Read Parameters from Calibration File~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Interpolates the Da parameters, energy and andgular range and aInner parameter from the tables and the retardation ratio
function Calculate_Da_values()

	SetDataFolder root:Data:tmpData:Analyser:

	NVAR LensMode, Ek, Ep, WF, aInner
	NVAR EkinLow, EkinHigh, AzimuthLow, AzimuthHigh
	Wave/T Calib_Matrix
	Wave eRange
	
	// retardation ratio
	Variable rr=(Ek-WF)/Ep, rr_inf, rr_factor
	String value
		
	// erange comes from the callibration file. It contains the range of energies around the pass energy
	// TODO: What pixels does this correspond to? Is it always correct, regardless of curved or straight entrance slit, and angle mode?
	EkinLow=Ek+eRange(0)*Ep
	EkinHigh=Ek+eRange(1)*Ep
	// This appears to be independent of retardation ratio.
	AzimuthLow = str2num(StringFromList(0,Calib_Matrix(LensMode)(0)(0)," "))
	AzimuthHigh = str2num(StringFromList(1,Calib_Matrix(LensMode)(0)(0)," "))
	
	Make/O/N=24 w_rr
	w_rr[]=Str2Num(Calib_Matrix(LensMode)(1+p)(0))
	// Will this work? we should look for NaNs...
	WaveStats/Q w_rr
	Redimension/N=(V_npnts) w_rr

	// find closest retardation ratio in table
	rr_inf=BinarySearch(w_rr, rr)
	// fraction from this to next retardation ratio in table
	rr_factor=BinarySearchInterp(w_rr, rr)-rr_inf
	
	// linear interpolation of the calibration values
	aInner =  Str2Num(Calib_Matrix(LensMode)(1+rr_inf)(1))*(1-rr_factor)+Str2Num(Calib_Matrix(LensMode)(1+rr_inf+1)(1))*rr_factor
	
	Make/O/N=3 Da1_value=0
	value=Calib_Matrix(LensMode)(1+rr_inf)(2)
	Da1_value[0] = str2num(StringFromList(0,value," "))*(1-rr_factor)
	Da1_value[1] = str2num(StringFromList(1,value," "))*(1-rr_factor)
	Da1_value[2] = str2num(StringFromList(2,value," "))*(1-rr_factor)
	value=Calib_Matrix(LensMode)(1+rr_inf+1)(2)
	Da1_value[0] += str2num(StringFromList(0,value," "))*(rr_factor)
	Da1_value[1] += str2num(StringFromList(1,value," "))*(rr_factor)
	Da1_value[2] += str2num(StringFromList(2,value," "))*(rr_factor)
	
	Make/O/N=3 Da3_value=0
	value=Calib_Matrix(LensMode)(1+rr_inf)(3)
	Da3_value[0] = str2num(StringFromList(0,value," "))*(1-rr_factor)
	Da3_value[1] = str2num(StringFromList(1,value," "))*(1-rr_factor)
	Da3_value[2] = str2num(StringFromList(2,value," "))*(1-rr_factor)
	value=Calib_Matrix(LensMode)(1+rr_inf+1)(3)
	Da3_value[0] += str2num(StringFromList(0,value," "))*(rr_factor)
	Da3_value[1] += str2num(StringFromList(1,value," "))*(rr_factor)
	Da3_value[2] += str2num(StringFromList(2,value," "))*(rr_factor)
	
	Make/O/N=3 Da5_value=0
	value=Calib_Matrix(LensMode)(1+rr_inf)(4)
	Da5_value[0] = str2num(StringFromList(0,value," "))*(1-rr_factor)
	Da5_value[1] = str2num(StringFromList(1,value," "))*(1-rr_factor)
	Da5_value[2] = str2num(StringFromList(2,value," "))*(1-rr_factor)
	value=Calib_Matrix(LensMode)(1+rr_inf+1)(4)
	Da5_value[0] += str2num(StringFromList(0,value," "))*(rr_factor)
	Da5_value[1] += str2num(StringFromList(1,value," "))*(rr_factor)
	Da5_value[2] += str2num(StringFromList(2,value," "))*(rr_factor)
	
	Make/O/N=3 Da7_value=0
	value=Calib_Matrix(LensMode)(1+rr_inf)(5)
	Da7_value[0] = str2num(StringFromList(0,value," "))*(1-rr_factor)
	Da7_value[1] = str2num(StringFromList(1,value," "))*(1-rr_factor)
	Da7_value[2] = str2num(StringFromList(2,value," "))*(1-rr_factor)
	value=Calib_Matrix(LensMode)(1+rr_inf+1)(5)
	Da7_value[0] += str2num(StringFromList(0,value," "))*(rr_factor)
	Da7_value[1] += str2num(StringFromList(1,value," "))*(rr_factor)
	Da7_value[2] += str2num(StringFromList(2,value," "))*(rr_factor)

end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function Calculate_Polynomial_Coef_Da()

	SetDataFolder root:Data:tmpData:'Analyser':

	NVAR Ek, Ep
	Wave eShift, Da1_value, Da3_value, Da5_value, Da7_value

	Make/O/N=3 Da_value_x, D1_coef, D3_coef, D5_coef, D7_coef
	
	// the calibration values are given for points at (-5% 0% +5%) of the pass energy, stored in eShift
	Da_value_x=eShift*Ep
	Da_value_x+=Ek
	// interpolate the Da waves with 3rd order polynomes
	CurveFit/Q/NTHR=0 poly 3, kwCWave=D1_coef,  Da1_value /X=Da_value_x 
	if( sum(Da3_value) != 0)
		CurveFit/Q/NTHR=0 poly 3, kwCWave=D3_coef,  Da3_value /X=Da_value_x
	else
		D3_coef=0
	endif
	if( sum(Da5_value) != 0)
		CurveFit/Q/NTHR=0 poly 3, kwCWave=D5_coef,  Da5_value /X=Da_value_x 
	else
		D5_coef=0
	endif
	if( sum(Da7_value) != 0)
		CurveFit/Q/NTHR=0 poly 3, kwCWave=D7_coef,  Da7_value /X=Da_value_x 
	else
		D7_coef=0
	endif
end


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// calculates the actual correction matrix
function Calculate_MatrixCorrection()

	SetDataFolder root:Data:tmpData:'Analyser':

	NVAR PixelSize, magnification, E_Offset_px, Ang_Offset_px, LensMode, Ek, Ep, De1, aInner
	NVAR EkinLow, EkinHigh, AzimuthLow, AzimuthHigh
	NVAR Binning
	NVAR Edge_pos, Edge_Slope
	
	String Str
	// for full frames, if cropped in program we need to do something else?
	Variable nx_pixel=1376/Binning, ny_pixel=1040/Binning
	Make/O/N=(nx_pixel, ny_pixel) MCP_Position_mm_Matrix=NaN, Angular_Correction=NaN
	Make/O/N=(nx_pixel) w_E, E_Correction=NaN
	Make/O/N=(ny_pixel) w_Ang

	Variable Edge_Coef
	
	// Just linear scaling according to the ranges from the calibration file. What limits are they done for?
	//w_E[] = EkinLow+p/(nx_pixel-1)*(EkinHigh-EkinLow)
	//w_Ang[] = AzimuthLow+p/(ny_pixel-1)*(AzimuthHigh-AzimuthLow)
	// Allow 20% more range
	//w_Ang[] = AzimuthLow*1.2+(p-Ang_Offset_px)/(ny_pixel-1)*(AzimuthHigh-AzimuthLow)*1.2
	// store positions in wave scaling of AngularCorrection now!
	setscale/P x, EkinLow, (EkinHigh-EkinLow)/nx_pixel, "eV", MCP_Position_mm_Matrix, Angular_Correction, E_Correction
	setscale/P y, AzimuthLow*1.2, (AzimuthHigh-AzimuthLow)*1.2/ny_pixel, "deg", MCP_Position_mm_Matrix, Angular_Correction

	// energy correction? What does it do? Seems to be strictly linear...
	// back-calculation of the pixel position from the energy in the w_E wave. Used for the proper interpolation later
	//E_Correction = limit( round( (w_E- Ek)/Ep/de1 / magnification/(PixelSize*Binning) + nx_pixel/2 + E_Offset_px)  , 0, nx_pixel)
	// no rounding for proper interpolation
	//E_Correction = (w_E- Ek)/Ep/de1 / magnification/(PixelSize*Binning) + nx_pixel/2 + E_Offset_px
	E_Correction = (x- Ek)/Ep/de1 / magnification/(PixelSize*Binning) + nx_pixel/2 + E_Offset_px
	// this is a Matrix with the mm-position for the energies and angles. This uses all the calibration parameters.
	MCP_Position_mm_Matrix = MCP_Position_mm( x, y )
	// Matrix that stores pixel position in the angular row for all energies and angles
	//Angular_Correction[][] = limit( round( MCP_Position_mm_Matrix(p)(q)/magnification/(PixelSize*Binning) + ny_pixel/2 + Ang_Offset_px)  , 0, ny_pixel)
	// do proper interpolation later
	Angular_Correction[][] = MCP_Position_mm_Matrix[p][q]/magnification/(PixelSize*Binning) + ny_pixel/2 + Ang_Offset_px
	
	Str="LensMode:;Ek:;PE:;Binning:"
	Str=ReplaceNumberByKey("LensMode", Str, LensMode)
	Str=ReplaceNumberByKey("Ek", Str, Ek)
	Str=ReplaceNumberByKey("PE", Str, Ep)
	Str=ReplaceNumberByKey("Binning", Str, binning)
	Note/K Angular_Correction, Str
	
	string temp = note(Angular_correction)
	// calculate the Jacobian determinant for Normalization
	duplicate/o Angular_Correction, Jacobian_Determinant, w_dxdE, w_dxdA, w_dydE, w_dydA
	// Energy derivative of angular correction
	differentiate/DIM=0 w_dydE
	// angular derivative of angular correction
	differentiate/DIM=1 w_dydA
	// energy derivative of energy correction
	duplicate/o E_correction, w_temp
	differentiate w_temp
	w_dxdE = w_temp[p]
	// angular derivative of energy correction is flat
	w_dxdA = 0
	Jacobian_Determinant = abs(w_dxdE*w_dydA - w_dydE*w_dxdA)
	killWaves/z w_dxdE, w_dxdA, w_dydE, w_dydA, w_temp
	
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// additional linear correction
function Scaling_Correction()

	SetDataFolder root:Data:tmpData:'Analyser':

	NVAR PixelSize, magnification, E_Offset_px, Ang_Offset_px, LensMode, Ek, Ep, De1, aInner
	NVAR EkinLow, EkinHigh, AzimuthLow, AzimuthHigh
	NVAR Binning
	NVAR Edge_pos, Edge_Slope
	
	Wave MCP_Position_mm_Matrix, w_E
	
	String Str
	Variable nx_pixel=1376/Binning, ny_pixel=1040/Binning
	//Make/O/N=(nx_pixel, ny_pixel) Angular_Correction=NaN
	Make/O/N=(nx_pixel) w_LinearCorrection=NaN
	setscale/P x, EkinLow, (EkinHigh-EkinLow)/nx_pixel, "eV", w_LinearCorrection
	wave Angular_Correction

	Variable Edge_Coef
	
	Edge_Coef = tan(Edge_Slope*pi/180)/MCP_Position_mm( Ek, Edge_pos)/Ek/Ep/De1
	w_LinearCorrection[]=1/(1+Edge_Coef*(x-Ek))

	//Angular_Correction[][] = limit( round( (w_LinearCorrection(p)*MCP_Position_mm_Matrix(p)(q) )/magnification/(PixelSize*Binning) + ny_pixel/2 + Ang_Offset_px)  , 0, ny_pixel)
	// No rounding for proper interpolation
	Angular_Correction[][] =w_LinearCorrection[p]*MCP_Position_mm_Matrix[p][q]/magnification/(PixelSize*Binning) + ny_pixel/2 + Ang_Offset_px
	
	//Str=note(Angular_Correction)
	//Str=ReplaceNumberByKey("LensMode", Str, LensMode)
	//Str=ReplaceNumberByKey("Ek", Str, Ek)
	//Str=ReplaceNumberByKey("PE", Str, Ep)
	//Note/K Angular_Correction, Str
	
	// calculate the Jacobian determinant for Normalization
	duplicate/o Angular_Correction, Jacobian_Determinant, w_dxdE, w_dxdA, w_dydE, w_dydA
	// Energy derivative of angular correction
	differentiate/DIM=0 w_dydE
	// angular derivative of angular correction
	differentiate/DIM=1 w_dydA
	// energy derivative of energy correction
	duplicate/o E_correction, w_temp
	differentiate w_temp
	w_dxdE = w_temp[p]
	// angular derivative of energy correction is flat
	w_dxdA = 0
	Jacobian_Determinant = abs(w_dxdE*w_dydA - w_dydE*w_dxdA)
	killWaves/z w_dxdE, w_dxdA, w_dydE, w_dydA, w_temp
	
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~
function MCP_Position_mm(Ek, Ang)
Variable Ek, Ang

	NVAR aInner=root:Data:tmpData:Analyser:aInner
	Wave D1=root:Data:tmpData:Analyser:D1_coef
	Wave D3=root:Data:tmpData:Analyser:D3_coef
	Wave D5=root:Data:tmpData:Analyser:D5_coef
	Wave D7=root:Data:tmpData:Analyser:D7_coef

	if( abs(Ang) <= aInner)
		return zInner(Ek, Ang)
	else
		return sign(Ang)*(zInner(Ek, aInner) + (abs(Ang)-aInner)*zInner_Diff(Ek, aInner))
	endif
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~
// returns distance of given angle from center of MCP in mm, at the given energy
function zInner(Ek, Ang)
Variable Ek, Ang

	Wave D1=root:Data:tmpData:Analyser:D1_coef
	Wave D3=root:Data:tmpData:Analyser:D3_coef
	Wave D5=root:Data:tmpData:Analyser:D5_coef
	Wave D7=root:Data:tmpData:Analyser:D7_coef

	return poly(D1, Ek )*(Ang) + 10^-2*poly(D3, Ek )*(Ang)^3 + 10^-4*poly(D5, Ek )*(Ang)^5 + 10^-6*poly(D7, Ek )*(Ang)^7
end

//~~~~~~~~~~~~~~~~~~~~~~~~~~~
function zInner_Diff(Ek, Ang)
Variable Ek, Ang

	Wave D1=root:Data:tmpData:Analyser:D1_coef
	Wave D3=root:Data:tmpData:Analyser:D3_coef
	Wave D5=root:Data:tmpData:Analyser:D5_coef
	Wave D7=root:Data:tmpData:Analyser:D7_coef

	return poly(D1, Ek ) + 3*10^-2*poly(D3, Ek )*(Ang)^2 + 5*10^-4*poly(D5, Ek )*(Ang)^4 + 7*10^-6*poly(D7, Ek )*(Ang)^6
end
