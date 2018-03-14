#pragma once

#include <cstddef>      // NULL
#include <iostream>
#include <map>


#define __FILENAME__ (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)


template< class T >
class Singleton
{
private:
    static T * mInstance;
public:
    static T & getInstance()
    {
        if ( mInstance == NULL )
            mInstance = new T();
        // only default value, because Error is supposed to be a fatal error from which we can't recover
        mInstance->activate( "", "Error" );
        return * mInstance;
    }
    virtual ~Singleton() { mInstance = 0; }
protected:
    Singleton(){} // can't be private, because it needs to be implicitly called from derived classes constructor?
    Singleton( Singleton const & ){}
};
// https://stackoverflow.com/questions/3229883/static-member-initialization-in-a-class-template
template < class T > T * Singleton<T>::mInstance = NULL;


class SelectiveLogger : public Singleton< SelectiveLogger >
{
friend class Singleton< SelectiveLogger >;
protected:
    // can't be private, because it needs to be implicitly called from derived classes constructor
    SelectiveLogger(){}
public:
    ~SelectiveLogger(){}

private:
    std::map< std::string, std::map< std::string, int > > mLevels;

public:
    /**
     * only returns true if logging level was explicitly set
     */
    bool isActive
    (
        std::string const & rFile,
        std::string const & rLevel
    ) const
    {
        auto const & it = mLevels.find( rFile );
        if ( it == mLevels.end() )
            return false;
        auto const it2 = it->second.find( rLevel );
        return it2 != it->second.end() && it2->second;
    };

    void activate  ( std::string const & f, std::string const & l, int const i = 1 ){ mLevels[f][l] = i; };
    void deactivate( std::string const & f, std::string const & l, int const i = 0 ){ mLevels[f][l] = i; };

    template< class T >
    SelectiveLogger & writeln
    (
        std::string const & rFile ,
        std::string const & rLevel,
        T           const & x
    )
    {
        if ( isActive( rFile, rLevel ) )
            std::cerr << x << "\n";
        return *this;
    }
};

/**
 * In contrast to SelectiveLogger this class is not a singleton and can
 * be instantiated e.g. for each file as a private member and can store
 * the file name for file-specific flags and easier access without getInstance
 */
class SelectedLogger : public Singleton< SelectedLogger >
{
private:
    std::string mActiveLevel;
    std::string mActiveFile ;

public:
    SelectedLogger
    (
        std::string const & rFile  = "",
        std::string const & rLevel = ""
    )
    {
        mActiveFile  = rFile ;
        mActiveLevel = rLevel;
    }

    inline bool isActive( std::string const & rLevel ) const
    { return SelectiveLogger::getInstance().isActive( mActiveFile, rLevel ); }
    inline bool isActive( std::string const & rFile, std::string const & rLevel ) const
    { return SelectiveLogger::getInstance().isActive( rFile, rLevel ); }

    inline void activate  ( std::string const & rLevel ){ SelectiveLogger::getInstance().activate  ( mActiveFile, rLevel ); };
    inline void deactivate( std::string const & rLevel ){ SelectiveLogger::getInstance().deactivate( mActiveFile, rLevel ); };

    inline bool isActive ( void ) const { return SelectiveLogger::getInstance().isActive( mActiveFile, mActiveLevel ); };
    inline void activate  ( void ){ SelectiveLogger::getInstance().activate  ( mActiveFile, mActiveLevel ); };
    inline void deactivate( void ){ SelectiveLogger::getInstance().deactivate( mActiveFile, mActiveLevel ); };

    inline SelectedLogger & file ( std::string const & s ){ mActiveFile  = s; return *this; };
    inline SelectedLogger & level( std::string const & s ){ mActiveLevel = s; return *this; };
    /* @todo maybe return a copy and only write out in destructor, like geniously done here?:
     * https://stackoverflow.com/questions/6168107/how-to-implement-a-good-debug-logging-feature-in-a-project */
    inline SelectedLogger & operator()( std::string const & f, std::string const & l ){  return file(f).level(l); }
    inline SelectedLogger & operator()( std::string const & l ){  return file( mActiveFile ).level(l); }

    /* Note that this does not work with std::endl @todo make it work by deriving from ostream? */
    /* Note !!! also regarding basically all other const & arguments ...
     * If used with a static constexpr member, this unwantedly leads to a
     * definition error, because even if it is inline ( const & ), an
     * address has to be taken, meaning the constexpr member needs to be
     * defined normally inside the body... which is normally not done nor
     * wanted! Note that things like std::string can't be declared as constexpr anyway,
     * so it does not have that problem */
    template< class T >
    inline SelectedLogger & operator<< ( T const x )
    {
        /* Currently this only uses the global level settings from
         * SelectiveLogger. In future this should use SelectiveLogger::writeln,
         * so that e.g. each line can be prepended with date, filename and
         * log level or so that everything can be (additionally) written into
         * a file
         * @todo make logger throw exception with error message
         * @todo predefine some values like error above ..
         */
        if ( SelectiveLogger::getInstance().isActive( mActiveFile, mActiveLevel ) )
            std::cerr << x;
        //else
        //    std::cerr << "[SelectiveLogger] " << mActiveFile << "::" << mActiveLevel << " is not activated!\n";
        return *this;
    }
};

/*
Example what could be inside a constructor:

    mLog.file( __FILENAME__ );
    mLog.activate( "Benchmark" ); // make and print timings. Might even execute things inside loops, thereby influencing performance. Shouldn't influence surrounding benchmarks, but if so, then you might wanna not only use true/false, but give specific numbers to activate and deactivate
    mLog.activate( "Check" );   // heavyweight time consuming checks which are not necessary, but might influence benchmarks
    mLog.activate( "Error" );   // Fatal error from which we can't or don't want to recover
    mLog.activate( "Info" );    // Lightweight info which doesn't consume time unlike checks, but still could spam the log full of messages
    mLog.activate( "Stat" );    // Similar to checks, but instead time-consuming statistics which are not influencing the program logic in any way and therefore are alway safe to deactivate unlike "Check" which you might want to keep activated normally
    mLog.activate( "Warning" ); // like "Info", but a bit more important. It makes sense to keep warning but deactivate Info, but not really vice-versa
    mLog( "Info" ) << "Initialized logging module.\n";
    if ( mLog.isActive( "Check" ) )
    {
        int nErrors = 0;
        for ( int i = 0u; i < 1000000; ++i )
            if ( ! mLog.isActive( "Check" ) )
                ++nErrors;
        mLog( "Info" ) << "Finished check for bitflips in memory\n";
        if ( nErrors > 0 )
        {
            std::stringstream msg;
            msg << "Memory seems to be bugged!";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
    }
*/
