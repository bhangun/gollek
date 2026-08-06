package tech.kayys.gollek.model.repo.kaggle;

import jakarta.enterprise.inject.Produces;
import jakarta.inject.Singleton;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import tech.kayys.alkhawarizm.spi.model.ModelRepository;

/**
 * CDI producer for the Kaggle repository.
 */
public class KaggleRepositoryProvider {

    @Produces
    @Singleton
    public ModelRepository kaggleRepository(KaggleClient client, KaggleConfig config) {
        return new KaggleRepository(client, config);
    }
}
