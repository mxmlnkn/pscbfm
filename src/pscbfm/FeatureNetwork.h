#pragma once


#include <LeMonADE/analyzer/AnalyzerWriteBfmFile.h>
#include <LeMonADE/feature/Feature.h>
#include <LeMonADE/io/AbstractRead.h>
#include <LeMonADE/io/AbstractWrite.h>
#include <LeMonADE/io/FileImport.h>


/**
 * @class NetworkTypeTag
 * @brief Extends monomers by a signed integer (int32_t) as tag and also
 *        adds a getter and setter method. This tag indicates the monomer type
 *          NONE(0), HEPARIN(1), STARPEG(2), or CROSSLINKER(3)
 *
 **/
class NetworkTypeTag
{
public:

    enum NETWORK_TYPE_TAG
    {
        NONE        = 0,    //!< Monomer is not specified
        HEPARIN     = 1,    //!< Monomer belongs to heparin
        STARPEG     = 2,    //!< Monomer belongs to starPEG
        CROSSLINKER = 3     //!< Monomer belongs to Cross-linker
    };

    //! Standard constructor- initially the tag is set to NULL.
    NetworkTypeTag():Tag(NONE){}

    //! Getting the tag of the monomer.
    inline NETWORK_TYPE_TAG getNetworkTypeTag() const { return Tag; }

    /**
     * @brief Setting the tag of the monomer with \para NetworkTypeTag.
     *
     * @param attr
     */
    inline void setNetworkTypeTag( NETWORK_TYPE_TAG _NetworkTypeTag ){ this->Tag = _NetworkTypeTag; }

private:
     //! Private variable holding the tag. Default is NULL.
    NETWORK_TYPE_TAG Tag;

};

class FeatureNetwork:public Feature
{
public:

    FeatureNetwork():NrOfStars(0), NrOfStarsOld(0),NrOfMonomersPerStarArm(29),NrOfCrosslinker(0), NrOfCrosslinkerOld(0), CrosslinkerFunctionality(0){};
    virtual ~FeatureNetwork() {
    }

    //! This Feature requires a monomer_extensions.
    typedef LOKI_TYPELIST_1(NetworkTypeTag) monomer_extensions;



    uint32_t getNrOfCrosslinker() const {
        return NrOfCrosslinker;
    }

    void setNrOfCrosslinker(uint32_t nrOfCrosslinker) {
        NrOfCrosslinker = nrOfCrosslinker;
    }

    uint32_t getNrOfMonomersPerStarArm() const {
        return NrOfMonomersPerStarArm;
    }

    void setNrOfMonomersPerStarArm(uint32_t nrOfMonomersPerStarArm) {
        NrOfMonomersPerStarArm = nrOfMonomersPerStarArm;
    }

    uint32_t getNrOfStars() const {
        return NrOfStars;
    }

    void setNrOfStars(uint32_t nrOfStars) {
        NrOfStars = nrOfStars;
    }

    uint32_t getCrosslinkerFunctionality() const {
        return CrosslinkerFunctionality;
    }

    void setCrosslinkerFunctionality(uint32_t crosslinkerFunctionality) {
        CrosslinkerFunctionality = crosslinkerFunctionality;
    }

    uint32_t getNrOfCrosslinkerOld() const {
        return NrOfCrosslinkerOld;
    }

    void setNrOfCrosslinkerOld(uint32_t nrOfCrosslinkerOld) {
        NrOfCrosslinkerOld = nrOfCrosslinkerOld;
    }

    uint32_t getNrOfStarsOld() const {
        return NrOfStarsOld;
    }

    void setNrOfStarsOld(uint32_t nrOfStarsOld) {
        NrOfStarsOld = nrOfStarsOld;
    }

    ;

    //We need 3 Features to handle the oscillatory forces
    //typedef LOKI_TYPELIST_2(FeatureAttributes,FeatureBoltzmann) required_features_back;

    //! For all unknown moves: this does nothing
    template<class IngredientsType>
    bool checkMove(const IngredientsType& ingredients,const MoveBase& move) const;

    //! Overloaded for moves of type MoveScMonomer to check for sinusoidal movement
    //template<class IngredientsType>
    //bool checkMove(const IngredientsType& ingredients,MoveLocalSc& move) const;

    //! Synchronize with system: check for right torque machines.
    template<class IngredientsType>
    void synchronize(IngredientsType& ingredients);

    //! Export the relevant functionality for reading bfm-files to the responsible reader object
    template <class IngredientsType>
    void exportRead(FileImport <IngredientsType>& fileReader);

     //! Export the relevant functionality for writing bfm-files to the responsible writer object
    template <class IngredientsType>
    void exportWrite(AnalyzerWriteBfmFile <IngredientsType>& fileWriter) const;

private:


    uint32_t NrOfStars; //number of Stars
    uint32_t NrOfStarsOld; //number of Stars in creational process
    uint32_t NrOfMonomersPerStarArm; //number OfMonomersPerStarArm
    uint32_t NrOfCrosslinker; //number of Crosslinker
    uint32_t NrOfCrosslinkerOld; //number of Crosslinker in creational process
    uint32_t CrosslinkerFunctionality; //chem. functionality of Crosslinker


};

/*****************************************************************/
/**
 * @class ReadNumStars
 *
 * @brief Handles BFM-File-Reads \b #!NrOfStars
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template < class IngredientsType>
class ReadNumStars: public ReadToDestination<IngredientsType>
{
public:
    ReadNumStars(IngredientsType& i):ReadToDestination<IngredientsType>(i){}
  virtual ~ReadNumStars(){}
  virtual void execute();
};

template<class IngredientsType>
void ReadNumStars<IngredientsType>::execute()
{
    std::cout<<"reading ReadNumStars...";

    uint32_t numStars = 0;
    IngredientsType& ingredients=this->getDestination();
    std::istream& source=this->getInputStream();

    std::string line;
    getline(source,line);
    numStars = atoi(line.c_str());
    std::cout << "#!NrOfStars" << (numStars) << std::endl;

    ingredients.setNrOfStars(numStars);
}




/*****************************************************************/
/**
 * @class WriteNumStars
 *
 * @brief Handles BFM-File-Write \b #!NrOfStars
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template <class IngredientsType>
class WriteNumStars:public AbstractWrite<IngredientsType>
{
public:
    WriteNumStars(const IngredientsType& i)
    :AbstractWrite<IngredientsType>(i){this->setHeaderOnly(true);}

    virtual ~WriteNumStars(){}

    virtual void writeStream(std::ostream& strm);
};

template<class IngredientsType>
void WriteNumStars<IngredientsType>::writeStream(std::ostream& stream)
{
    stream<<"#!NrOfStars=" << (this->getSource().getNrOfStars()) << std::endl<< std::endl;
}



/*****************************************************************/
/**
 * @class ReadNumStarsOld
 *
 * @brief Handles BFM-File-Reads \b #!NrOfStarsOld
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template < class IngredientsType>
class ReadNumStarsOld: public ReadToDestination<IngredientsType>
{
public:
    ReadNumStarsOld(IngredientsType& i):ReadToDestination<IngredientsType>(i){}
  virtual ~ReadNumStarsOld(){}
  virtual void execute();
};

template<class IngredientsType>
void ReadNumStarsOld<IngredientsType>::execute()
{
    std::cout<<"reading ReadNumStarsOld...";

    uint32_t numStarsOld = 0;
    IngredientsType& ingredients=this->getDestination();
    std::istream& source=this->getInputStream();

    std::string line;
    getline(source,line);
    numStarsOld = atoi(line.c_str());
    std::cout << "#!NrOfStarsOld" << (numStarsOld) << std::endl;

    ingredients.setNrOfStarsOld(numStarsOld);
}




/*****************************************************************/
/**
 * @class WriteNumStarsOld
 *
 * @brief Handles BFM-File-Write \b #!NrOfStarsOld
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template <class IngredientsType>
class WriteNumStarsOld:public AbstractWrite<IngredientsType>
{
public:
    WriteNumStarsOld(const IngredientsType& i)
    :AbstractWrite<IngredientsType>(i){this->setHeaderOnly(true);}

    virtual ~WriteNumStarsOld(){}

    virtual void writeStream(std::ostream& strm);
};

template<class IngredientsType>
void WriteNumStarsOld<IngredientsType>::writeStream(std::ostream& stream)
{
    stream<<"#!NrOfStarsOld=" << (this->getSource().getNrOfStarsOld()) << std::endl<< std::endl;
}



/*****************************************************************/
/**
 * @class ReadNumStarArm
 *
 * @brief Handles BFM-File-Reads \b #!NrOfMonomersPerStarArm
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template < class IngredientsType>
class ReadNumStarArm: public ReadToDestination<IngredientsType>
{
public:
    ReadNumStarArm(IngredientsType& i):ReadToDestination<IngredientsType>(i){}
  virtual ~ReadNumStarArm(){}
  virtual void execute();
};

template<class IngredientsType>
void ReadNumStarArm<IngredientsType>::execute()
{
    std::cout<<"reading ReadNumStarArm...";

    uint32_t numStarArm = 0;
    IngredientsType& ingredients=this->getDestination();
    std::istream& source=this->getInputStream();

    std::string line;
    getline(source,line);
    numStarArm = atoi(line.c_str());
    std::cout << "#!NrOfMonomersPerStarArm" << (numStarArm) << std::endl;

    ingredients.setNrOfMonomersPerStarArm(numStarArm);
}




/*****************************************************************/
/**
 * @class WriteNumStarArm
 *
 * @brief Handles BFM-File-Write \b #!NrOfMonomersPerStarArm
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template <class IngredientsType>
class WriteNumStarArm:public AbstractWrite<IngredientsType>
{
public:
    WriteNumStarArm(const IngredientsType& i)
    :AbstractWrite<IngredientsType>(i){this->setHeaderOnly(true);}

    virtual ~WriteNumStarArm(){}

    virtual void writeStream(std::ostream& strm);
};

template<class IngredientsType>
void WriteNumStarArm<IngredientsType>::writeStream(std::ostream& stream)
{
    stream<<"#!NrOfMonomersPerStarArm=" << (this->getSource().getNrOfMonomersPerStarArm()) << std::endl<< std::endl;
}


/*****************************************************************/
/**
 * @class ReadNumCrosslinker
 *
 * @brief Handles BFM-File-Reads \b #!NrOfCrosslinker
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template < class IngredientsType>
class ReadNumCrosslinker: public ReadToDestination<IngredientsType>
{
public:
    ReadNumCrosslinker(IngredientsType& i):ReadToDestination<IngredientsType>(i){}
  virtual ~ReadNumCrosslinker(){}
  virtual void execute();
};

template<class IngredientsType>
void ReadNumCrosslinker<IngredientsType>::execute()
{
    std::cout<<"reading ReadNumCrosslinker...";

    uint32_t numCrosslinker = 0;
    IngredientsType& ingredients=this->getDestination();
    std::istream& source=this->getInputStream();

    std::string line;
    getline(source,line);
    numCrosslinker = atoi(line.c_str());
    std::cout << "#!NrOfCrosslinker=" << (numCrosslinker) << std::endl;

    ingredients.setNrOfCrosslinker(numCrosslinker);
}




/*****************************************************************/
/**
 * @class WriteNumCrosslinker
 *
 * @brief Handles BFM-File-Write \b #!NrOfCrosslinker
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template <class IngredientsType>
class WriteNumCrosslinker:public AbstractWrite<IngredientsType>
{
public:
    WriteNumCrosslinker(const IngredientsType& i)
    :AbstractWrite<IngredientsType>(i){this->setHeaderOnly(true);}

    virtual ~WriteNumCrosslinker(){}

    virtual void writeStream(std::ostream& strm);
};

template<class IngredientsType>
void WriteNumCrosslinker<IngredientsType>::writeStream(std::ostream& stream)
{
    stream<<"#!NrOfCrosslinker=" << (this->getSource().getNrOfCrosslinker()) << std::endl<< std::endl;
}

/*****************************************************************/
/**
 * @class ReadNumCrosslinkerOld
 *
 * @brief Handles BFM-File-Reads \b #!NrOfCrosslinkerOld
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template < class IngredientsType>
class ReadNumCrosslinkerOld: public ReadToDestination<IngredientsType>
{
public:
    ReadNumCrosslinkerOld(IngredientsType& i):ReadToDestination<IngredientsType>(i){}
  virtual ~ReadNumCrosslinkerOld(){}
  virtual void execute();
};

template<class IngredientsType>
void ReadNumCrosslinkerOld<IngredientsType>::execute()
{
    std::cout<<"reading ReadNumCrosslinkerOld...";

    uint32_t numCrosslinkerOld = 0;
    IngredientsType& ingredients=this->getDestination();
    std::istream& source=this->getInputStream();

    std::string line;
    getline(source,line);
    numCrosslinkerOld = atoi(line.c_str());
    std::cout << "#!NrOfCrosslinkerOld=" << (numCrosslinkerOld) << std::endl;

    ingredients.setNrOfCrosslinkerOld(numCrosslinkerOld);
}




/*****************************************************************/
/**
 * @class WriteNumCrosslinkerOld
 *
 * @brief Handles BFM-File-Write \b #!NrOfCrosslinkerOld
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template <class IngredientsType>
class WriteNumCrosslinkerOld:public AbstractWrite<IngredientsType>
{
public:
    WriteNumCrosslinkerOld(const IngredientsType& i)
    :AbstractWrite<IngredientsType>(i){this->setHeaderOnly(true);}

    virtual ~WriteNumCrosslinkerOld(){}

    virtual void writeStream(std::ostream& strm);
};

template<class IngredientsType>
void WriteNumCrosslinkerOld<IngredientsType>::writeStream(std::ostream& stream)
{
    stream<<"#!NrOfCrosslinkerOld=" << (this->getSource().getNrOfCrosslinkerOld()) << std::endl<< std::endl;
}

/*****************************************************************/
/**
 * @class ReadFuncCrosslinker
 *
 * @brief Handles BFM-File-Reads \b #!CrosslinkerFunctionality
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template < class IngredientsType>
class ReadFuncCrosslinker: public ReadToDestination<IngredientsType>
{
public:
    ReadFuncCrosslinker(IngredientsType& i):ReadToDestination<IngredientsType>(i){}
  virtual ~ReadFuncCrosslinker(){}
  virtual void execute();
};

template<class IngredientsType>
void ReadFuncCrosslinker<IngredientsType>::execute()
{
    std::cout<<"reading ReadFuncCrosslinker...";

    uint32_t numFuncCrosslinker = 0;
    IngredientsType& ingredients=this->getDestination();
    std::istream& source=this->getInputStream();

    std::string line;
    getline(source,line);
    numFuncCrosslinker = atoi(line.c_str());
    std::cout << "#!CrosslinkerFunctionality" << (numFuncCrosslinker) << std::endl;

    ingredients.setCrosslinkerFunctionality(numFuncCrosslinker);
}




/*****************************************************************/
/**
 * @class WriteFuncCrosslinker
 *
 * @brief Handles BFM-File-Write \b #!NrOfCrosslinker
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template <class IngredientsType>
class WriteFuncCrosslinker:public AbstractWrite<IngredientsType>
{
public:
    WriteFuncCrosslinker(const IngredientsType& i)
    :AbstractWrite<IngredientsType>(i){this->setHeaderOnly(true);}

    virtual ~WriteFuncCrosslinker(){}

    virtual void writeStream(std::ostream& strm);
};

template<class IngredientsType>
void WriteFuncCrosslinker<IngredientsType>::writeStream(std::ostream& stream)
{
    stream<<"#!CrosslinkerFunctionality=" << (this->getSource().getCrosslinkerFunctionality()) << std::endl<< std::endl;
}


/**
 * @details The function is called by the Ingredients class when an object of type Ingredients
 * is associated with an object of type FileImport. The export of the Reads is thus
 * taken care automatically when it becomes necessary.\n
 * Registered Read-In Commands:
 * * #!NrOfStars
 * * #!NrOfMonomersPerStarArm
 * * #!NrOfCrosslinker
 * * #!CrosslinkerFunctionality
 * * #!NrOfStarsOld
 * * #!NrOfCrosslinkerOld
 *
 *
 * @param fileReader File importer for the bfm-file
 * @param destination List of Feature to write-in from the read values.
 * @tparam IngredientsType Ingredients class storing all system information.
 **/
template<class IngredientsType>
void FeatureNetwork::exportRead(FileImport< IngredientsType >& fileReader)
{
   // fileReader.registerRead("#!electric_force_osci", new ReadElectricForceOsci<FeatureNetwork>(*this));


    fileReader.registerRead("#!NrOfStars", new ReadNumStars<FeatureNetwork>(*this));
    fileReader.registerRead("#!NrOfStarsOld", new ReadNumStarsOld<FeatureNetwork>(*this));
    fileReader.registerRead("#!NrOfMonomersPerStarArm",new ReadNumStarArm<FeatureNetwork>(*this));
    fileReader.registerRead("#!NrOfCrosslinker", new ReadNumCrosslinker<FeatureNetwork>(*this));
    fileReader.registerRead("#!CrosslinkerFunctionality", new ReadFuncCrosslinker<FeatureNetwork>(*this));
    fileReader.registerRead("#!NrOfCrosslinkerOld", new ReadNumCrosslinkerOld<FeatureNetwork>(*this));
}

/**
 * The function is called by the Ingredients class when an object of type Ingredients
 * is associated with an object of type AnalyzerWriteBfmFile. The export of the Writes is thus
 * taken care automatically when it becomes necessary.\n
 * Registered Write-Out Commands:
 * * #!NrOfStars
 * * #!NrOfMonomersPerStarArm
 * * #!NrOfCrosslinker
 * * #!CrosslinkerFunctionality
 * * #!NrOfStarsOld
 * * #!NrOfCrosslinkerOld
 *
 *
 * @param fileWriter File writer for the bfm-file.
 */
template<class IngredientsType>
void FeatureNetwork::exportWrite(AnalyzerWriteBfmFile< IngredientsType >& fileWriter) const
{
    fileWriter.registerWrite("#!NrOfStars=", new WriteNumStars<FeatureNetwork>(*this));
    fileWriter.registerWrite("#!NrOfStarsOld=", new WriteNumStarsOld<FeatureNetwork>(*this));
    fileWriter.registerWrite("#!NrOfMonomersPerStarArm=",new WriteNumStarArm<FeatureNetwork>(*this));
    fileWriter.registerWrite("#!NrOfCrosslinker=", new WriteNumCrosslinker<FeatureNetwork>(*this));
    fileWriter.registerWrite("#!CrosslinkerFunctionality=", new WriteFuncCrosslinker<FeatureNetwork>(*this));
    fileWriter.registerWrite("#!NrOfCrosslinkerOld=", new WriteNumCrosslinkerOld<FeatureNetwork>(*this));

    //fileWriter.registerWrite("#!electric_force_osci",new WriteElectricForceOsci<IngredientsType>(fileWriter.getIngredients_()));
}



template<class IngredientsType>
bool FeatureNetwork::checkMove(const IngredientsType& ingredients, const MoveBase& move) const
{
    return true;
}


/**
 * @details Synchronizes the Torques and check for consistency of the system - esp if connections are ok.
 *
 * @param ingredients A reference to the IngredientsType - mainly the system.
 */
template<class IngredientsType>
void FeatureNetwork::synchronize(IngredientsType& ingredients)
        {

    //synchronize function of FeatureNetwork
    std::cout << "FeatureNetwork::synchronizing...";


    const typename IngredientsType::molecules_type& molecules=ingredients.getMolecules();

    for (size_t i=0; i< molecules.size(); ++i)
    {
        for (size_t j=0; j< molecules.getNumLinks(i); ++j){

            uint n = molecules.getNeighborIdx(i,j);

            // if the NetworkTypeTag is the same - this is maybe an error
            if (molecules[n].getNetworkTypeTag() == molecules[i].getNetworkTypeTag())
            {
                if((molecules[i].getNetworkTypeTag() == NetworkTypeTag::HEPARIN) && (molecules[n].getNetworkTypeTag() == NetworkTypeTag::HEPARIN))
                {
                    std::ostringstream errorMessage;
                    errorMessage << "FeatureNetwork::synchronize(): Invalid connection between monomer " << i << " at " << molecules[i];
                    errorMessage << " as type HEPARIN " << molecules[i].getNetworkTypeTag() << " and " << n << " at " <<  molecules[n];
                    errorMessage << " as type HEPARIN " << molecules[n].getNetworkTypeTag() << ".\n";

                    throw std::runtime_error(errorMessage.str());

                }

                if((molecules[i].getNetworkTypeTag() == NetworkTypeTag::STARPEG) && (molecules[n].getNetworkTypeTag() == NetworkTypeTag::STARPEG))
                {
                    std::ostringstream errorMessage;
                    errorMessage << "FeatureNetwork::synchronize(): Invalid connection between monomer " << i << " at " << molecules[i];
                    errorMessage << " as type STARPEG " << molecules[i].getNetworkTypeTag() << " and " << n << " at " <<  molecules[n];
                    errorMessage << " as type STARPEG " << molecules[n].getNetworkTypeTag() << ".\n";

                    throw std::runtime_error(errorMessage.str());

                }

                if((molecules[i].getNetworkTypeTag() == NetworkTypeTag::CROSSLINKER) && (molecules[n].getNetworkTypeTag() == NetworkTypeTag::CROSSLINKER))
                {
                    std::ostringstream errorMessage;
                    errorMessage << "FeatureNetwork::synchronize(): Invalid connection between monomer " << i << " at " << molecules[i];
                    errorMessage << " as type STARPEG " << molecules[i].getNetworkTypeTag() << " and " << n << " at " <<  molecules[n];
                    errorMessage << " as type STARPEG " << molecules[n].getNetworkTypeTag() << ".\n";

                    throw std::runtime_error(errorMessage.str());

                }

            }
            else
            {
                if((molecules[i].getNetworkTypeTag() == NetworkTypeTag::HEPARIN) && (molecules[n].getNetworkTypeTag() == NetworkTypeTag::CROSSLINKER))
                {
                    std::ostringstream errorMessage;
                    errorMessage << "FeatureNetwork::synchronize(): Invalid connection between monomer " << i << " at " << molecules[i];
                    errorMessage << " as type HEPARIN " << molecules[i].getNetworkTypeTag() << " and " << n << " at " <<  molecules[n];
                    errorMessage << " as type CROSSLINKER " << molecules[n].getNetworkTypeTag() << ".\n";

                    throw std::runtime_error(errorMessage.str());

                }

                if((molecules[i].getNetworkTypeTag() == NetworkTypeTag::CROSSLINKER) && (molecules[n].getNetworkTypeTag() == NetworkTypeTag::HEPARIN))
                {
                    std::ostringstream errorMessage;
                    errorMessage << "FeatureNetwork::synchronize(): Invalid connection between monomer " << i << " at " << molecules[i];
                    errorMessage << " as type CROSSLINKER " << molecules[i].getNetworkTypeTag() << " and " << n << " at " <<  molecules[n];
                    errorMessage << " as type HEPARIN " << molecules[n].getNetworkTypeTag() << ".\n";

                    throw std::runtime_error(errorMessage.str());

                }
            }

        }
    }

    for (size_t i=0; i< molecules.size(); ++i)
        {
            if(molecules[i].getNetworkTypeTag() == NetworkTypeTag::CROSSLINKER)
                if(molecules.getNumLinks(i) > CrosslinkerFunctionality)
                {
                    std::ostringstream errorMessage;
                    errorMessage << "FeatureNetwork::synchronize(): Invalid connection on Crosslinker " << i << " at " << molecules[i] << std::endl;
                    errorMessage << "Number of links on Crosslinker " << molecules.getNumLinks(i) << " , but should be restricted to " << CrosslinkerFunctionality;
                    errorMessage << ". Exiting...\n";

                    throw std::runtime_error(errorMessage.str());
                }
        }

    std::cout << "done\n";
}
